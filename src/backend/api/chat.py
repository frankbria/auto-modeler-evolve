import json
import re
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from chat.orchestrator import (
    build_system_prompt,
    detect_state,
    generate_suggestions,
    get_next_step_chips,
)
from core.query_engine import generate_chart_for_message
from db import get_session
from models.conversation import Conversation
from models.dataset import Dataset
from models.deployment import Deployment
from models.feature_set import FeatureSet
from models.model_run import ModelRun
from models.prediction_log import PredictionLog
from models.project import Project

router = APIRouter(prefix="/api/chat", tags=["chat"])

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

# Keywords that trigger an inline model readiness assessment
_READINESS_PATTERNS = re.compile(
    r"\b(ready|readiness|production.ready|deploy|is.*(model|it).*ready|can.*deploy|"
    r"should.*deploy|good enough|production|go.live|ship it|launch)\b",
    re.IGNORECASE,
)

# Keywords that trigger an inline drift assessment
_DRIFT_PATTERNS = re.compile(
    r"\b(drift|drifting|shifted|predictions.*(off|wrong|different|changed)|"
    r"still accurate|performance.*(drop|degraded|worse)|data.*(changed|shifted|stale)|"
    r"retrain|re.train|model.*stale|stale.*model|distribution.*changed)\b",
    re.IGNORECASE,
)

# Keywords that trigger a model health + retraining guidance
_HEALTH_PATTERNS = re.compile(
    r"\b(model.health|health.*model|how.*model.*doing|model.*status|"
    r"should.*retrain|time.*retrain|need.*retrain|when.*retrain|"
    r"train.*again|update.*model|refresh.*model|model.*up.?to.?date|"
    r"model.*current|model.*stale|stale.*model|model.*fresh|"
    r"is.*model.*still.*good|check.*model.*health)\b",
    re.IGNORECASE,
)

# Keywords that trigger a hyperparameter tuning suggestion
_TUNE_PATTERNS = re.compile(
    r"\b(tune|tuning|optimize|optimise|improve.*model|better.*model|model.*better|"
    r"increase accuracy|boost performance|hyperparameter|grid search|random search|"
    r"can.*do better|make.*better|improve.*accuracy|improve.*performance|"
    r"best.*hyperparameter|find.*best.*param)\b",
    re.IGNORECASE,
)

# Keywords that trigger the model improvement advisor card
# Distinct from _TUNE_PATTERNS (hyperparameter-only) — these ask for broad advice
_IMPROVEMENT_PATTERNS = re.compile(
    r"\b(how.*improve.*model|what.*improve|improve.*prediction|"
    r"how.*make.*model.*better|how.*get.*better.*result|suggestion.*model|"
    r"model.*suggestion|advice.*model|model.*advice|"
    r"what.*wrong.*model|why.*model.*poor|model.*not.*good|"
    r"how.*increase.*r2|how.*increase.*accuracy|how.*boost|"
    r"what.*should.*do.*model|next.*step.*model|model.*improvement|"
    r"improvement.*suggestion|any.*suggestion|give.*suggestion)\b",
    re.IGNORECASE,
)

# Goal-driven training: analyst sets a target metric and AutoModeler tries algorithms
_GOAL_TRAIN_PATTERNS = re.compile(
    r"(?:"
    r"(?:i\s+need|i\s+want|we\s+need|we\s+want)\s+(?:at\s+least\s+)?(?:\d+(?:\.\d+)?%|\d*\.\d+)\s+(?:accuracy|f1|r.?2|r-squared|precision)\b|"
    r"(?:reach|hit|achieve|get\s+to|target|aim\s+for)\s+(?:\d+(?:\.\d+)?%|\d*\.\d+)\s+(?:accuracy|f1|r.?2|r-squared|precision)\b|"
    r"train\s+(?:a\s+)?model\s+(?:until|to)\s+(?:it\s+)?(?:reach(?:es)?|hits?|gets?|achieves?)\b|"
    r"keep\s+trying\s+(?:different\s+)?(?:models?|algorithms?)\s+(?:until|to)\b|"
    r"(?:try|test)\s+(?:different|all|multiple|various)\s+(?:models?|algorithms?)\s+(?:to\s+)?(?:find|reach|hit|get)\s+\d+\b|"
    r"goal.driven\s+training\b|"
    r"(?:train|build)\s+(?:a\s+)?model\s+(?:that\s+)?(?:reaches?|hits?|achieves?)\s+\d+\b|"
    r"automatic(?:ally)?\s+(?:find|train|try)\s+(?:the\s+)?best\s+(?:model|algorithm)\s+(?:to|for|that)\b"
    r")",
    re.IGNORECASE,
)

_GOAL_METRIC_RE = re.compile(
    r"\b(accuracy|f1(?:\s+score)?|r.?2|r-squared|r\s+squared|precision|recall)\b",
    re.IGNORECASE,
)


def _extract_goal_target(message: str, problem_type: str) -> tuple[str, float] | None:
    """Extract (goal_metric, goal_target) from a natural-language message.

    Examples handled:
        "I need 85% accuracy"  → ("accuracy", 0.85)
        "reach 0.90 R²"        → ("r2", 0.90)
        "hit 80% F1"           → ("f1", 0.80)

    Returns None if no numeric threshold is found.
    """
    # Determine metric
    metric_match = _GOAL_METRIC_RE.search(message)
    if metric_match:
        raw = metric_match.group(1).lower().replace(" ", "_")
        if "r" in raw and ("2" in raw or "squared" in raw):
            metric = "r2"
        elif "f1" in raw:
            metric = "f1"
        elif "precision" in raw:
            metric = "precision"
        elif "recall" in raw:
            metric = "recall"
        else:
            metric = "accuracy"
    else:
        metric = "r2" if problem_type == "regression" else "accuracy"

    # Try percentage (e.g. "85%")
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", message)
    if pct_match:
        val = float(pct_match.group(1)) / 100.0
        if 0 < val <= 1.0:
            return metric, round(val, 4)

    # Try plain decimal (e.g. "0.85")
    dec_match = re.search(r"\b(0\.\d{1,4})\b", message)
    if dec_match:
        val = float(dec_match.group(1))
        if 0 < val <= 1.0:
            return metric, round(val, 4)

    return None


# Keywords that trigger model selection advisor
# Distinct from _IMPROVEMENT_PATTERNS (improve existing) — these ask "which model to use"
_MODEL_SELECT_PATTERNS = re.compile(
    r"\b(which model.*use|what model.*use|pick.*best.*model|pick.*model|"
    r"recommend.*model|which model.*best|best model.*me|"
    r"most.*explain|explain.*model|easy.*explain|"
    r"most.*accurate.*model|highest.*accuracy.*model|"
    r"most.*stable|most.*consistent.*model|"
    r"fastest.*model|quickest.*model|low.*latency.*model|"
    r"model.*my.*goal|choose.*model|select.*model.*for|compare.*model.*criteria|"
    r"which.*algorithm.*use|what.*algorithm.*best)\b",
    re.IGNORECASE,
)

# Keywords that trigger a cross-deployment alerts scan
# Note: no trailing \b — patterns use .* wildcards so plurals ("alerts") work fine
_ALERTS_PATTERNS = re.compile(
    r"\b(any.*alert|alert.*model|monitor|check.*model|model.*issue|problem.*model|"
    r"system.*status|health.*check|model.*health.*all|all.*model.*health|"
    r"\bissues\b|anything.*wrong|something.*wrong|model.*ok|models.*ok|"
    r"status.*update|how.*model.*doing|all.*deployment)",
    re.IGNORECASE,
)

# Keywords that trigger the model version history card
_HISTORY_PATTERNS = re.compile(
    r"\b(version.*histor|model.*histor|show.*histor|past.*run|previous.*run|"
    r"training.*histor|model.*over.*time|how.*model.*improv|model.*progress|"
    r"show.*improvement|histor.*model|how.*improv|trend.*model|"
    r"model.*trend|improving.*over|getting.*better)",
    re.IGNORECASE,
)

# Keywords that trigger the prediction analytics card
_ANALYTICS_PATTERNS = re.compile(
    r"\b(prediction.*analytic|analytic.*prediction|how.*many.*prediction|"
    r"prediction.*count|usage.*stat|stat.*usage|prediction.*volume|"
    r"prediction.*log|log.*prediction|how.*often.*predict|prediction.*traffic|"
    r"show.*analytic|prediction.*usage|usage.*dashboard)",
    re.IGNORECASE,
)

# Keywords that suggest a data cleaning operation
# Note: these suggest the operation; actual application requires user confirmation via button.
_CLEAN_PATTERNS = re.compile(
    r"\b(clean|fix.*missing|fill.*missing|fill.*null|fill.*empty|"
    r"remove.*duplicat|drop.*duplicat|deduplic|dedup|"
    r"remove.*rows.*where|drop.*rows.*where|filter.*out|exclude.*rows|"
    r"cap.*outlier|remove.*outlier|handle.*outlier|clip.*outlier|"
    r"drop.*column|remove.*column|delete.*column|"
    r"fix.*data|clean.*data|clean.*up|data.*quality|improve.*data|"
    r"missing.*value|null.*value|handle.*null|handle.*missing)",
    re.IGNORECASE,
)

_FILL_COL_PATTERN = re.compile(
    r"\bfill\s+(?:missing\s+)?(?:values?\s+in\s+)?[\"']?(\w+)[\"']?\s+"
    r"(?:column\s+)?with\s+(mean|median|mode|zero|[\d.]+)",
    re.IGNORECASE,
)
_FILTER_COL_PATTERN = re.compile(
    r"\b(?:remove|drop|filter|exclude)\s+rows?\s+where\s+[\"']?(\w+)[\"']?\s*"
    r"(>|<|>=|<=|==|!=|is|equals?|greater than|less than|not)\s*([\d.]+|\w+)",
    re.IGNORECASE,
)
_CAP_COL_PATTERN = re.compile(
    r"\bcap\s+(?:outliers?\s+in\s+)?[\"']?(\w+)[\"']?(?:\s+at\s+([\d.]+)\s*%?)?",
    re.IGNORECASE,
)
_DROP_COL_PATTERN = re.compile(
    r"\b(?:drop|remove|delete)\s+(?:the\s+)?(?:column\s+)?[\"']?(\w+)[\"']?\s+column",
    re.IGNORECASE,
)

_OP_MAP = {
    "is": "eq",
    "equals": "eq",
    "equal": "eq",
    ">": "gt",
    "<": "lt",
    ">=": "gte",
    "<=": "lte",
    "!=": "ne",
    "not": "ne",
    "greater than": "gt",
    "less than": "lt",
}


def _detect_clean_op(message: str, columns: list[str]) -> dict | None:
    """Try to extract a specific cleaning operation from the user's message.

    Returns a dict matching CleanRequest fields, or None if only general intent
    detected (caller emits a generic suggestion).
    """
    col_set = set(c.lower() for c in columns)

    # fill_missing: "fill missing age with median" / "fill age column with 0"
    m = _FILL_COL_PATTERN.search(message)
    if m:
        col_raw, strat = m.group(1), m.group(2).lower()
        col = next((c for c in columns if c.lower() == col_raw.lower()), col_raw)
        try:
            val = float(strat)
            return {
                "operation": "fill_missing",
                "column": col,
                "strategy": "value",
                "fill_value": val,
            }
        except ValueError:
            pass
        if strat in ("mean", "median", "mode", "zero"):
            return {"operation": "fill_missing", "column": col, "strategy": strat}

    # remove_duplicates: "remove duplicates"
    if re.search(r"\b(duplicate|dedup)\b", message, re.IGNORECASE):
        return {"operation": "remove_duplicates"}

    # filter_rows: "drop rows where quantity < 0"
    m = _FILTER_COL_PATTERN.search(message)
    if m:
        col_raw, op_raw, val_raw = m.group(1), m.group(2).strip().lower(), m.group(3)
        col = next((c for c in columns if c.lower() == col_raw.lower()), col_raw)
        operator = _OP_MAP.get(op_raw, "eq")
        try:
            val: float | str = float(val_raw)
        except ValueError:
            val = val_raw
        return {
            "operation": "filter_rows",
            "column": col,
            "operator": operator,
            "value": val,
        }

    # cap_outliers: "cap outliers in sales" / "cap revenue outliers at 99%"
    m = _CAP_COL_PATTERN.search(message)
    if m:
        col_raw = m.group(1)
        pct_raw = m.group(2)
        col = next((c for c in columns if c.lower() == col_raw.lower()), None)
        if col or col_raw.lower() in col_set:
            resolved = col or col_raw
            pct = float(pct_raw) if pct_raw else 99.0
            return {"operation": "cap_outliers", "column": resolved, "percentile": pct}

    # drop_column: "drop the sales column" / "remove region column"
    m = _DROP_COL_PATTERN.search(message)
    if m:
        col_raw = m.group(1)
        col = next((c for c in columns if c.lower() == col_raw.lower()), None)
        if col:
            return {"operation": "drop_column", "column": col}

    return None  # general cleaning intent — no specific op detected


# Keywords that suggest the user has new data to upload (guided refresh flow)
_REFRESH_PATTERNS = re.compile(
    r"\b(new data|new.*csv|updated.*data|updated.*csv|fresh.*data|"
    r"refresh.*data|refresh.*dataset|replace.*data|replace.*dataset|"
    r"upload.*new|new.*upload|latest.*data|have.*new.*file|"
    r"data.*changed|data.*updated|new.*version.*data|"
    r"re.?upload|update.*my.*data|new.*spreadsheet|new.*file)\b",
    re.IGNORECASE,
)

# Keywords that trigger anomaly detection on the current dataset
_ANOMALY_PATTERNS = re.compile(
    r"\b(anomal|unusual.*record|outlier|strange.*data|weird.*record|"
    r"suspicious|find.*weird|anything.*odd|odd.*row|odd.*record|"
    r"which.*record.*unusual|unusual.*data|data.*unusual|"
    r"detect.*anomal|spot.*anomal|identify.*anomal|show.*anomal)",
    re.IGNORECASE,
)

# Keywords that trigger cross-tabulation / pivot table analysis
_CROSSTAB_PATTERNS = re.compile(
    r"\b(break.*down|breakdown|cross.*tab|crosstab|pivot|"
    r"show.*by.*and|split.*by|across.*and|"
    r"by.*and.*by|group.*by.*and|"
    r"matrix.*of|comparison.*table|compare.*across|"
    r"revenue.*by.*region|sales.*by.*region|"
    r"(\w+)\s+by\s+(\w+)\s+and\s+(\w+))\b",
    re.IGNORECASE,
)

# Regex to extract "VALUE by ROW and COL" or "break down VALUE by ROW and COL"
_CROSSTAB_EXTRACT = re.compile(
    r"\b(?:break.*?down\s+)?(?:show\s+(?:me\s+)?)?(\w+)\s+by\s+(\w+)"
    r"(?:\s+and\s+(\w+))?",
    re.IGNORECASE,
)


def _detect_crosstab_request(message: str, columns: list[str]) -> dict | None:
    """Extract row_col, col_col, value_col from a crosstab message.

    Matches patterns like:
      "break down revenue by region and product"
      "show sales across channel and quarter"
      "pivot revenue by region"

    Returns dict with row_col, col_col, value_col, or None if not enough columns found.
    """
    col_lower = {c.lower(): c for c in columns}

    m = _CROSSTAB_EXTRACT.search(message)
    if not m:
        return None

    # Try to match each captured group against real column names
    groups = [g for g in m.groups() if g]
    matched_cols = [col_lower[g.lower()] for g in groups if g.lower() in col_lower]

    if len(matched_cols) < 2:
        return None

    # Heuristic: if we have 3 matches and first is numeric-sounding, it's the value
    if len(matched_cols) >= 3:
        value_col, row_col, col_col = matched_cols[0], matched_cols[1], matched_cols[2]
    else:
        # 2 matched: treat first as row, second as col; use count aggregation
        row_col, col_col = matched_cols[0], matched_cols[1]
        value_col = None

    return {"row_col": row_col, "col_col": col_col, "value_col": value_col}


# Keywords that trigger a computed/derived column suggestion
_COMPUTE_PATTERNS = re.compile(
    r"\b(add\s+(?:a\s+)?(?:new\s+)?(?:column|field|variable|metric)|"
    r"create\s+(?:a\s+)?(?:new\s+)?(?:column|field|variable|metric)|"
    r"calculate\s+(?:a\s+)?(?:new\s+)?(?:column|field)?|"
    r"compute\s+(?:a\s+)?(?:new\s+)?(?:column|field)?|"
    r"derive\s+(?:a\s+)?(?:new\s+)?(?:column|field)?|"
    r"make\s+(?:a\s+)?(?:new\s+)?(?:column|field)|"
    r"new\s+column\s+(?:called|named)|"
    r"column\s+called\s+\w+\s*=|"
    r"\w+\s*=\s*\w+\s*[+\-*/]\s*\w+)\b",
    re.IGNORECASE,
)

# Extract: "add column called NAME = EXPRESSION" / "create X as Y / Z"
_COMPUTE_EXTRACT = re.compile(
    r"(?:add|create|calculate|compute|derive|make)\s+(?:a\s+)?(?:new\s+)?"
    r"(?:column|field|variable|metric)?\s*"
    r"(?:called|named|as)?\s*['\"]?(\w+)['\"]?\s*"
    r"(?:=|as|equals?|that\s+is|which\s+is|\:)\s*(.+?)(?:\s*$)",
    re.IGNORECASE,
)


def _detect_compute_request(message: str, columns: list[str]) -> dict | None:
    """Extract column name and expression from a natural-language compute request.

    Matches patterns like:
      "add a column called margin = revenue / cost"
      "create profit_per_unit as profit / units"
      "calculate growth_rate = (sales - prev_sales) / prev_sales"

    Returns dict with {name, expression} or None if not parseable.
    """
    m = _COMPUTE_EXTRACT.search(message)
    if not m:
        return None

    name = m.group(1).strip()
    expression = m.group(2).strip()

    # Expression must reference at least one existing column
    col_lower = {c.lower() for c in columns}
    expr_words = set(re.findall(r"[a-zA-Z_]\w*", expression))
    if not expr_words.intersection(col_lower):
        return None

    return {"name": name, "expression": expression}


_COMPARE_PATTERNS = re.compile(
    r"\b("
    r"compare\s+\w[\w\s]*\s+(?:vs\.?|versus|and|with)\s+\w|"
    r"\w[\w\s]*\s+vs\.?\s+\w[\w\s]*|"
    r"\w[\w\s]*\s+versus\s+\w[\w\s]*|"
    r"difference\s+between\s+\w[\w\s]*\s+and\s+\w|"
    r"how\s+(?:does|do|is|are)\s+\w[\w\s]*\s+(?:compare|differ)\s+(?:to|from|with)\s+\w|"
    r"contrast\s+\w[\w\s]*\s+(?:and|with)\s+\w"
    r")\b",
    re.IGNORECASE,
)

# Extract two group terms: "compare X vs Y" / "X versus Y" / "difference between X and Y"
_COMPARE_EXTRACT = re.compile(
    r"(?:"
    r"compare\s+(.+?)\s+(?:vs\.?|versus|and|with)\s+(.+?)"
    r"|(.+?)\s+vs\.?\s+(.+?)"
    r"|(.+?)\s+versus\s+(.+?)"
    r"|difference\s+between\s+(.+?)\s+and\s+(.+?)"
    r"|how\s+(?:does|do|is|are)\s+(.+?)\s+(?:compare|differ)\s+(?:to|from|with)\s+(.+?)"
    r")\s*(?:\?|$)",
    re.IGNORECASE,
)


def _detect_compare_request(message: str, df) -> dict | None:
    """Extract two group values and the column they come from.

    Searches the DataFrame for a categorical column whose unique values
    contain both extracted terms. Returns {group_col, val1, val2} or None.
    """
    m = _COMPARE_EXTRACT.search(message.strip())
    if not m:
        return None

    # Pick the first two non-None groups (comes from alternation in pattern)
    groups = [g for g in m.groups() if g is not None]
    if len(groups) < 2:
        return None

    raw1 = groups[0].strip().rstrip("?").strip()
    raw2 = groups[1].strip().rstrip("?").strip()

    # For each categorical column, check if both terms exist in actual values
    for col in df.columns:
        if not (
            str(df[col].dtype) in ("object", "string", "str", "category")
            or df[col].nunique() <= 30
        ):
            continue
        vals_lower = {str(v).strip().lower() for v in df[col].dropna().unique()}
        if raw1.lower() in vals_lower and raw2.lower() in vals_lower:
            return {"group_col": col, "val1": raw1, "val2": raw2}

    return None


_FORECAST_PATTERNS = re.compile(
    r"(?:"
    r"forecast|predict\s+(?:the\s+)?next|project\s+(?:the\s+)?(?:next|future)|"
    r"what\s+will\s+\w+\s+be\s+(?:next|in)|"
    r"next\s+\d+\s+(?:day|week|month|quarter|period)|"
    r"future\s+(?:trend|value|predict)|extrapolate|"
    r"how\s+will\s+\w+\s+(?:grow|change|trend)|"
    r"revenue\s+(?:for\s+)?next|sales\s+(?:for\s+)?next"
    r")",
    re.IGNORECASE,
)

# Extract period count: "next 3 months", "forecast 6 weeks", "predict next 12 quarters"
_FORECAST_PERIODS_RE = re.compile(
    r"(?:next\s+|forecast\s+|predict\s+(?:the\s+)?next\s+)?(\d+)\s+"
    r"(day|week|month|quarter|period)s?",
    re.IGNORECASE,
)


def _detect_forecast_request(message: str) -> dict:
    """Extract forecast parameters from a natural-language message.

    Returns a dict with:
      - periods: int (default 6)
      - period_unit: str (default 'period')
    """
    m = _FORECAST_PERIODS_RE.search(message)
    if m:
        periods = min(24, max(1, int(m.group(1))))
        period_unit = m.group(2).lower()
    else:
        periods = 6
        period_unit = "period"
    return {"periods": periods, "period_unit": period_unit}


# Keywords that trigger target correlation analysis ("what drives X", "correlated with Y")
_CORRELATION_TARGET_PATTERNS = re.compile(
    r"(?:"
    r"what(?:'s|\s+is)?\s+(?:correlated|correlat(?:e|es|ing))\s+with|"
    r"what\s+(?:drives?|influences?|affects?|impacts?|predicts?)\s+\w|"
    r"which\s+(?:columns?|features?|variables?|factors?)\s+(?:correlate|relate|affect|drive|predict)|"
    r"show\s+(?:me\s+)?correlations?\s+(?:for|with|of)\s+\w|"
    r"correlat(?:e|es|ions?)\s+(?:for|with|of)\s+\w|"
    r"factors?\s+(?:affecting|influencing|driving|that\s+(?:affect|influence|drive))|"
    r"what\s+(?:is|are)\s+(?:related|correlated)\s+to"
    r")",
    re.IGNORECASE,
)


def _detect_correlation_target_request(
    message: str, df_columns: list[str]
) -> str | None:
    """Extract the target column name from a correlation request.

    Scans known DataFrame column names against the user's message (case-insensitive).
    Returns the first matching column name, or None if no column is mentioned.
    """
    msg_lower = message.lower()
    for col in df_columns:
        if col.lower() in msg_lower:
            return col
    return None


# Keywords that trigger a data readiness assessment (distinct from model readiness)
_DATA_READINESS_PATTERNS = re.compile(
    r"(?:"
    r"is\s+(?:my|the|this)\s+data\s+(?:ready|good|ok|clean|prepared|usable)|"
    r"data\s+(?:ready|readiness|quality\s+score|prepared|good\s+enough)|"
    r"can\s+(?:I|we)\s+(?:start\s+)?(?:train|model|build|use)|"
    r"ready\s+to\s+(?:train|model|build)|"
    r"is\s+(?:my|the|this)\s+data\s+(?:clean\s+enough|good\s+enough)|"
    r"check\s+(?:my\s+)?data|data\s+check|assess\s+(?:my\s+)?data|"
    r"data\s+(?:suitable|fit\s+for\s+modeling|fit\s+for\s+training)"
    r")",
    re.IGNORECASE,
)

_GROUP_PATTERNS = re.compile(
    r"(?:"
    r"(?:by|per|for\s+each|group\s+(?:by|on))\s+\w+|"
    r"break(?:down|s)?\s+(?:by|per|on)|"
    r"breakdown\s+(?:of\s+)?\w+\s+by|"
    r"(?:group|bucket|split|segment)\s+(?:by|on|into)|"
    r"(?:show|give)\s+me\s+\w+\s+by\s+\w+|"
    r"total\s+\w+\s+(?:by|per)\s+\w+|"
    r"(?:sum|average|count|mean)\s+(?:\w+\s+)?(?:by|per|for\s+each)\s+\w+"
    r")",
    re.IGNORECASE,
)


def _detect_group_request(message: str, df) -> dict | None:
    """Extract group_by column, optional value columns, and agg from message.

    Scans actual DataFrame column names for mentions in the message (case-insensitive).
    Returns a dict with 'group_col', 'value_cols', 'agg', or None if not enough
    columns can be identified.
    """
    lower = message.lower()
    cols = [c for c in df.columns]
    lower_to_col = {c.lower(): c for c in cols}
    lower_to_col.update({c.lower().replace("_", " "): c for c in cols})

    mentioned: list[str] = [lower_to_col[lc] for lc in lower_to_col if lc in lower]
    # Remove duplicates while preserving order
    seen: set[str] = set()
    mentioned_unique: list[str] = []
    for c in mentioned:
        if c not in seen:
            seen.add(c)
            mentioned_unique.append(c)

    if not mentioned_unique:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Try to identify the group-by column (categorical) and value columns (numeric)
    group_col: str | None = None
    value_cols: list[str] = []

    for c in mentioned_unique:
        if c in cat_cols and group_col is None:
            group_col = c
        elif c in numeric_cols:
            value_cols.append(c)

    # If only numeric columns mentioned and no categorical, try to find a
    # categorical column referenced by name in the message
    if group_col is None:
        for c in cat_cols:
            if c.lower() in lower or c.lower().replace("_", " ") in lower:
                group_col = c
                break

    if group_col is None:
        return None

    # Detect aggregation keyword
    agg = "sum"
    if re.search(r"\b(?:average|mean|avg)\b", lower):
        agg = "mean"
    elif re.search(r"\b(?:count|how\s+many|number\s+of)\b", lower):
        agg = "count"
    elif re.search(r"\bmin(?:imum)?\b", lower):
        agg = "min"
    elif re.search(r"\bmax(?:imum)?\b", lower):
        agg = "max"
    elif re.search(r"\bmedian\b", lower):
        agg = "median"

    return {
        "group_col": group_col,
        "value_cols": value_cols or None,
        "agg": agg,
    }


# Keywords that trigger showing the full correlation matrix heatmap
_HEATMAP_PATTERNS = re.compile(
    r"(?:"
    r"show\s+(?:me\s+)?(?:the\s+)?correlation\s+(?:matrix|heatmap|map)|"
    r"correlation\s+(?:matrix|heatmap|map)|"
    r"heatmap|"
    r"how\s+are\s+(?:my\s+)?(?:columns?|variables?|features?)\s+(?:correlated|related)|"
    r"show\s+(?:me\s+)?(?:the\s+)?correlations?\s+(?:between|among|across)\s+(?:all|my|the)|"
    r"all.?vs.?all\s+correlations?|"
    r"pairwise\s+correlations?|"
    r"full\s+correlation"
    r")",
    re.IGNORECASE,
)

# Keywords that trigger a column rename suggestion
_RENAME_PATTERNS = re.compile(
    r"(?:"
    r"rename\s+(?:column\s+)?['\"]?\w+['\"]?\s+(?:to|as)\s+['\"]?\w+|"
    r"call\s+(?:the\s+)?(?:column\s+)?['\"]?\w+['\"]?\s+['\"]?\w+|"
    r"change\s+(?:the\s+)?(?:column\s+)?name\s+(?:of\s+)?['\"]?\w+['\"]?\s+to\s+['\"]?\w+|"
    r"rename\s+['\"]?\w+['\"]?\s+column"
    r")",
    re.IGNORECASE,
)

# Extract: "rename X to Y" / "rename X as Y" / "change name of X to Y"
_RENAME_EXTRACT = re.compile(
    r"(?:"
    r"rename\s+(?:column\s+|the\s+)?['\"]?(\w+)['\"]?\s+(?:to|as)\s+['\"]?(\w+)['\"]?|"
    r"change\s+(?:the\s+)?(?:column\s+)?name\s+(?:of\s+)?['\"]?(\w+)['\"]?\s+to\s+['\"]?(\w+)['\"]?"
    r")",
    re.IGNORECASE,
)


def _detect_rename_request(message: str, columns: list[str]) -> dict | None:
    """Extract old_name and new_name from a rename request.

    Matches patterns like:
      "rename revenue_usd to Revenue"
      "rename the column rev_q1 to Q1 Revenue"
      "change the name of sales to total_sales"

    Returns dict with {old_name, new_name} where old_name matches an actual
    column (case-insensitive), or None if not parseable.
    """
    m = _RENAME_EXTRACT.search(message.strip())
    if not m:
        return None

    groups = [g for g in m.groups() if g is not None]
    if len(groups) < 2:
        return None

    raw_old, raw_new = groups[0].strip(), groups[1].strip()

    # Case-insensitive match against actual columns
    col_lower = {c.lower(): c for c in columns}
    old_name = col_lower.get(raw_old.lower())
    if not old_name:
        return None

    return {"old_name": old_name, "new_name": raw_new}


# Keywords that trigger a two-column scatter plot ("plot X vs Y")
_SCATTER_PATTERNS = re.compile(
    r"(?:"
    r"(?:scatter|plot|chart|graph)\s+(?:\w+\s+)?(?:vs\.?|versus|against|and)\s+\w+|"
    r"\bplot\s+(?:the\s+)?relationship\s+between|"
    r"\bscatter\s+plot\b|"
    r"\b(?:show|visualize|display)\s+(?:me\s+)?(?:the\s+)?relationship\s+between|"
    r"\brelationship\s+between\s+\w+\s+and\s+\w+|"
    r"\bhow\s+(?:does|do)\s+\w+\s+(?:relate|correlate)\s+to\s+\w+|"
    r"\b\w+\s+(?:vs\.?|versus|against)\s+\w+\s+(?:scatter|plot|chart|graph)"
    r")",
    re.IGNORECASE,
)


def _detect_scatter_request(message: str, df: "pd.DataFrame") -> dict | None:
    """Extract x_col and y_col from a scatter plot request.

    Uses column-name-first approach: scans known column names around separator
    words (vs, against, versus, and, between/and).

    Returns dict with {x_col, y_col} or None if two numeric columns not found.
    """
    numeric_cols = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    if len(numeric_cols) < 2:
        return None

    msg_lower = message.lower()

    # Helper: find best matching column in a text fragment (longest match first)
    def _match_col(fragment: str) -> str | None:
        fragment = fragment.strip()
        for col in sorted(df.columns, key=len, reverse=True):
            if col.lower() in fragment:
                return col
        return None

    # Pattern 1: "X vs Y", "X versus Y", "X against Y"
    for sep_pat in [r"\bvs\.?\b", r"\bversus\b", r"\bagainst\b"]:
        m = re.search(
            rf"(\w[\w\s]{{0,30}}?)\s+{sep_pat}\s+(\w[\w\s]{{0,30}}?)(?:\s+(?:scatter|plot|chart|graph|$)|$)",
            msg_lower,
        )
        if m:
            x_col = _match_col(m.group(1))
            y_col = _match_col(m.group(2))
            if x_col and y_col and x_col != y_col:
                return {"x_col": x_col, "y_col": y_col}

    # Pattern 2: "between X and Y"
    m = re.search(
        r"\bbetween\s+(\w[\w\s]{0,30}?)\s+and\s+(\w[\w\s]{0,30}?)(?:\s*[.?!]|$)",
        msg_lower,
    )
    if m:
        x_col = _match_col(m.group(1))
        y_col = _match_col(m.group(2))
        if x_col and y_col and x_col != y_col:
            return {"x_col": x_col, "y_col": y_col}

    # Fallback: find all numeric columns mentioned in message, use first two
    mentioned = []
    for col in sorted(df.columns, key=len, reverse=True):
        if col.lower() in msg_lower and col in numeric_cols and col not in mentioned:
            mentioned.append(col)
    if len(mentioned) >= 2:
        return {"x_col": mentioned[0], "y_col": mentioned[1]}

    return None


# Keywords that trigger an automated data story / full analysis
_STORY_PATTERNS = re.compile(
    r"\b("
    r"analyze\s+(?:my\s+)?(?:data|dataset|this)|"
    r"walk\s+(?:me\s+)?through\s+(?:my\s+|this\s+)?(?:data|dataset)|"
    r"give\s+(?:me\s+)?(?:a\s+|the\s+)?(?:full|complete|comprehensive|overall)\s+(?:analysis|summary|overview|picture)|"
    r"what(?:'?s|\s+is)\s+interesting|"
    r"what\s+(?:should\s+i\s+know|are\s+the\s+(?:key\s+)?(?:insights?|findings?|highlights?))|"
    r"summarize\s+(?:my\s+)?(?:data|dataset|this)|"
    r"(?:tell|show)\s+me\s+(?:everything|the\s+story|the\s+key\s+facts?)|"
    r"data\s+(?:story|overview|summary)|"
    r"(?:full|complete)\s+(?:data\s+)?analysis|"
    r"what\s+do\s+you\s+(?:see|find)\s+in\s+(?:my\s+)?(?:data|this)"
    r")\b",
    re.IGNORECASE,
)

# Keywords that trigger chat-initiated model training
_TRAIN_PATTERNS = re.compile(
    r"(?:"
    r"(?:train|build|create|fit|run|start)\s+(?:a\s+)?(?:new\s+)?(?:model|predictor|classifier|regressor)|"
    r"(?:start|begin|kick.?off)\s+training|"
    r"train\s+(?:me\s+)?(?:a\s+)?model\s+(?:to\s+)?predict|"
    r"build\s+(?:a\s+)?(?:ml|machine.learning|predictive|ai)\s+model|"
    r"(?:I\s+want\s+to|let'?s|can\s+you)\s+(?:train|build|fit|model)\s+(?:a\s+)?(?:model|predictor)|"
    r"model\s+(?:this|the|my)\s+(?:data|dataset)|"
    r"predict\s+\w+\s+(?:with|using)\s+(?:a\s+)?(?:model|ml|machine\s+learning)"
    r")",
    re.IGNORECASE,
)

_TIME_SPLIT_PATTERNS = re.compile(
    r"(?:"
    r"(?:use|enable|switch\s+to|apply)\s+(?:a\s+)?(?:time.?based|chronological|temporal|date.?based)\s+split|"
    r"(?:time.?based|chronological|temporal)\s+(?:train.?test\s+)?split|"
    r"split\s+(?:by|on|using)\s+(?:date|time|chronolog)|"
    r"train\s+on\s+(?:older|historical|past|earlier)\s+data|"
    r"test\s+on\s+(?:newer|recent|future|later)\s+data|"
    r"(?:respect|preserve|use)\s+(?:the\s+)?(?:date|time|temporal)\s+order|"
    r"time\s+series\s+split|"
    r"(?:use\s+)?random\s+split"
    r")",
    re.IGNORECASE,
)

# Extract "predict X" / "target is X" / "model for X" — scan against known columns
_TRAIN_TARGET_EXTRACT = re.compile(
    r"(?:"
    r"predict\s+(?:the\s+)?['\"]?(\w+)['\"]?|"
    r"target\s+(?:is\s+|column\s+(?:is\s+)?)?['\"]?(\w+)['\"]?|"
    r"(?:model|forecast|estimate)\s+(?:the\s+)?['\"]?(\w+)['\"]?"
    r")",
    re.IGNORECASE,
)


def _detect_selection_criteria(message: str) -> str:
    """Detect analyst criteria intent from a model selection message.

    Returns one of: accuracy | explainability | stability | speed | balanced
    """
    msg = message.lower()

    # Explainability keywords
    if any(
        kw in msg
        for kw in [
            "explain",
            "transparent",
            "interpretable",
            "understand",
            "stakeholder",
            "simple",
            "easy to",
        ]
    ):
        return "explainability"

    # Accuracy keywords
    if any(
        kw in msg
        for kw in [
            "accurate",
            "accuracy",
            "precise",
            "best performance",
            "highest",
            "best metric",
            "most predict",
        ]
    ):
        return "accuracy"

    # Speed / latency keywords
    if any(
        kw in msg
        for kw in [
            "fast",
            "quick",
            "speed",
            "latency",
            "real-time",
            "real time",
            "low latency",
            "high volume",
        ]
    ):
        return "speed"

    # Stability / consistency keywords
    if any(
        kw in msg
        for kw in [
            "stable",
            "consistent",
            "reliable",
            "trust",
            "robust",
        ]
    ):
        return "stability"

    # Default: balanced
    return "balanced"


def _detect_train_target(message: str, df_columns: list[str]) -> str | None:
    """Extract a target column from a training request message.

    First tries pattern extraction ("predict X"), then scans known column names.
    Returns the matched column name (original casing) or None.
    """
    col_lower = {c.lower(): c for c in df_columns}

    # Try pattern-based extraction first
    for m in _TRAIN_TARGET_EXTRACT.finditer(message):
        for g in m.groups():
            if g and g.lower() in col_lower:
                return col_lower[g.lower()]

    # Fallback: any column name mentioned in the message
    msg_lower = message.lower()
    for col in df_columns:
        if col.lower() in msg_lower:
            return col

    return None


# Keywords that trigger a non-destructive data filter
_FILTER_PATTERNS = re.compile(
    r"\b("
    r"filter\s+(?:to|by|the\s+data|my\s+data)|"
    r"focus\s+on\s+(?:only\s+)?(?:the\s+)?\w|"
    r"narrow\s+(?:down\s+)?(?:to|the)|"
    r"show\s+(?:me\s+)?only|"
    r"just\s+(?:look\s+at|show(?:\s+me)?)|"
    r"look\s+at\s+only|"
    r"limit\s+(?:to|the\s+data\s+to)|"
    r"subset\s+(?:the\s+data\s+)?(?:to|by|where)|"
    r"restrict\s+(?:to|the\s+data\s+to)|"
    r"only\s+(?:consider|include|use)\s+\w|"
    r"where\s+\w+\s+(?:is|=|>|<|>=|<=|contains?)\s+\w|"
    r"for\s+(?:the\s+)?\w+\s+(?:region|segment|category|group|quarter|year)|"
    r"set\s+(?:a\s+)?filter|"
    r"last\s+\d+\s+(?:day|week|month|year)s?|"
    r"(?:this|last)\s+(?:year|month|quarter)|"
    r"q[1-4](?:\s+20\d{2})?|"
    r"(?:first|second|third|fourth)\s+quarter|"
    r"show\s+(?:20\d{2}|(?:january|february|march|april|may|june|july|august|september|october|november|december))\s"
    r")\b",
    re.IGNORECASE,
)

# Keywords to clear an active filter
_CLEAR_FILTER_PATTERNS = re.compile(
    r"\b("
    r"clear\s+(?:the\s+)?filter|remove\s+(?:the\s+)?filter|"
    r"reset\s+(?:the\s+)?filter|turn\s+off\s+(?:the\s+)?filter|"
    r"show\s+all\s+(?:data|rows)|no\s+filter|"
    r"full\s+dataset|all\s+(?:the\s+)?data(?!\s+story)"
    r")\b",
    re.IGNORECASE,
)

# Keywords that trigger model card / explain-my-model
_MODEL_CARD_PATTERNS = re.compile(
    r"\b("
    r"explain\s+(?:my\s+|the\s+)?model|"
    r"what\s+does\s+(?:my\s+|the\s+)?model\s+do|"
    r"how\s+does\s+(?:my\s+|the\s+)?model\s+work|"
    r"tell\s+me\s+about\s+(?:my\s+|the\s+)?model|"
    r"describe\s+(?:my\s+|the\s+)?model|"
    r"model\s+(?:summary|overview|card|report|explanation)|"
    r"how\s+good\s+is\s+(?:my\s+|the\s+)?model|"
    r"summarize\s+(?:my\s+|the\s+)?model|"
    r"what\s+(?:drives|influences|affects)\s+(?:my\s+)?predictions"
    r")\b",
    re.IGNORECASE,
)

# Keywords that trigger chat-initiated model deployment
_DEPLOY_CHAT_PATTERNS = re.compile(
    r"\b("
    r"deploy\s+(?:my\s+)?(?:best\s+|selected\s+|the\s+)?model|"
    r"(?:go|make\s+(?:it|the\s+model))\s+live|"
    r"publish\s+(?:my\s+|the\s+)?model|"
    r"launch\s+(?:my\s+|the\s+)?(?:model|api)|"
    r"put\s+(?:my\s+|the\s+)?model\s+(?:in\s+)?(?:production|prod)|"
    r"create\s+(?:an?\s+)?(?:api|endpoint)\s+(?:for|from)\s+(?:my\s+|the\s+)?model|"
    r"ship\s+(?:my\s+|the\s+)?model|"
    r"share\s+(?:my\s+|the\s+)?model\s+(?:as\s+)?(?:an?\s+)?(?:api|link)|"
    r"make\s+(?:a\s+)?(?:prediction\s+)?(?:api|endpoint)"
    r")\b",
    re.IGNORECASE,
)

# Keywords that trigger chat-initiated PDF report download
_REPORT_PATTERNS = re.compile(
    r"\b("
    r"generate\s+(?:a\s+)?(?:model\s+)?(?:pdf\s+)?report|"
    r"create\s+(?:a\s+)?(?:model\s+)?(?:pdf\s+)?report|"
    r"(?:download|export|get)\s+(?:a\s+|the\s+)?(?:model\s+)?(?:pdf\s+)?report|"
    r"(?:make|build)\s+(?:a\s+)?(?:pdf\s+|model\s+)?report|"
    r"give\s+me\s+(?:a\s+)?(?:pdf\s+|model\s+)?report|"
    r"(?:pdf|model)\s+report|"
    r"report\s+(?:for\s+(?:my\s+|the\s+)?model|download|pdf)|"
    r"share\s+(?:a\s+)?(?:model\s+)?report|"
    r"print\s+(?:a\s+|the\s+)?(?:model\s+)?report"
    r")\b",
    re.IGNORECASE,
)

# Keywords that trigger chat-driven feature engineering suggestions
_FEATURE_SUGGEST_PATTERNS = re.compile(
    r"(?i)\b("
    r"suggest\s+(?:some\s+)?(?:feature|features|transformation|transformations)|"
    r"recommend\s+(?:some\s+)?(?:feature|features|transformation|transformations)|"
    r"(?:what|which)\s+features?\s+(?:should|can|could|would)|"
    r"feature\s+engineering|"
    r"(?:improve|engineer|prepare|build|create)\s+(?:my\s+)?features?|"
    r"(?:show|list)\s+(?:me\s+)?(?:feature|possible)\s+(?:suggestions?|transforms?)|"
    r"help\s+(?:me\s+)?(?:with\s+)?(?:feature|features)|"
    r"(?:any|what)\s+(?:feature|transform)(?:ation)?\s+(?:suggestions?|ideas?)"
    r")\b",
    re.IGNORECASE,
)

# Keywords that trigger applying all feature engineering suggestions
_FEATURE_APPLY_PATTERNS = re.compile(
    r"(?i)\b("
    r"apply\s+(?:all\s+)?(?:the\s+)?(?:feature\s+)?(?:suggestions?|transforms?|engineering|features?)|"
    r"(?:accept|use|approve)\s+(?:all\s+)?(?:the\s+)?(?:feature\s+)?(?:suggestions?|transforms?)|"
    r"yes,?\s+apply\s+(?:all\s+)?(?:the\s+)?(?:feature\s+)?(?:suggestions?|transforms?)|"
    r"do\s+(?:all\s+)?(?:the\s+)?(?:feature\s+)?(?:engineering|transforms?|suggestions?)|"
    r"run\s+(?:the\s+)?feature\s+(?:engineering|transforms?)"
    r")\b",
    re.IGNORECASE,
)

# Model performance by segment — "how does my model perform by region?"
_SEGMENT_PERF_PATTERNS = re.compile(
    r"(?i)\b("
    r"how\s+(?:does|do)\s+(?:my\s+)?model\s+(?:perform|work|do)\s+(?:by|across|for|on|per)|"
    r"model\s+(?:performance|accuracy|error|score)\s+(?:by|across|per|for)|"
    r"(?:performance|accuracy)\s+(?:by|across|per|breakdown\s+by)|"
    r"(?:performance|accuracy)\s+(?:breakdown|split)|"
    r"(?:check|analyze|show)\s+model\s+(?:performance|accuracy)\s+(?:by|across|per)|"
    r"does\s+my\s+model\s+(?:work|perform)\s+(?:equally|the\s+same|consistently)|"
    r"which\s+(?:segment|group|category|region|product|class)\s+(?:does\s+my\s+model\s+)?(?:perform|work)s?\s+(?:worst|best|poorly|well)"
    r")\b",
    re.IGNORECASE,
)


def _detect_segment_perf_col(message: str, df: "pd.DataFrame") -> str | None:
    """Extract the column name from a segment performance request.

    Scans the message for a word that matches a low-cardinality column in the DataFrame.
    Falls back to the first categorical column if no match.
    """
    import re as _re

    msg_lower = message.lower()
    candidates = []
    for col in df.columns:
        if df[col].nunique() <= 30 and df[col].nunique() >= 2:
            candidates.append(col)

    # Try to find a column name mentioned in the message
    for col in candidates:
        if col.lower() in msg_lower:
            return col
        # Also try individual words from the column name (for multi-word columns)
        for part in _re.split(r"[\s_-]", col.lower()):
            if len(part) > 2 and part in msg_lower:
                return col

    # Fall back to first categorical column (likely the most useful grouper)
    return candidates[0] if candidates else None


def _load_working_df(
    file_path: "Path", active_filter_conditions: list | None
) -> "pd.DataFrame":
    """Load a DataFrame from CSV and apply active filter conditions if any."""
    df = pd.read_csv(file_path)
    if active_filter_conditions:
        from core.filter_view import apply_active_filter

        df = apply_active_filter(df, active_filter_conditions)
    return df


# Keywords that trigger a column profile deep-dive card
_COLUMN_PROFILE_PATTERNS = re.compile(
    r"(?i)("
    r"tell\s+me\s+(?:more\s+)?about\s+(?:the\s+)?(?:column\s+)?(?:\w+\s+)?column|"
    r"profile\s+(?:the\s+)?(?:column\s+)?|"
    r"describe\s+(?:the\s+)?(?:column\s+)?|"
    r"analyze\s+(?:the\s+)?(?:column\s+)?|"
    r"what\s+(?:is|does|are)\s+(?:the\s+)?(?:column\s+)?(?:\w+\s+)?(?:column\s+)?(?:look\s+like|contain|show|represent)|"
    r"show\s+(?:me\s+)?(?:the\s+)?stats\s+(?:for|of|on)\s+|"
    r"(?:distribution|histogram|breakdown)\s+(?:of|for)\s+|"
    r"deep.?dive\s+(?:into|on)\s+|"
    r"explore\s+(?:the\s+)?(?:column\s+)?|"
    r"what\s+are\s+the\s+values\s+(?:in|of)\s+"
    r")",
    re.IGNORECASE,
)


def _detect_profile_col(message: str, df: "pd.DataFrame") -> str | None:
    """Extract the column name to profile from the message.

    Scans the message for a word matching any column name in the DataFrame.
    Returns the first match, or None if no column is identified.
    """
    import re as _re

    msg_lower = message.lower()
    # Try exact column name match first
    for col in df.columns:
        if col.lower() in msg_lower:
            return col
    # Try partial word match (handles snake_case column names)
    for col in df.columns:
        for part in _re.split(r"[\s_-]", col.lower()):
            if len(part) > 2 and part in msg_lower:
                return col
    return None


# Keywords that trigger K-means clustering / natural segmentation
_CLUSTER_PATTERNS = re.compile(
    r"(?i)\b("
    r"cluster\s+(?:my\s+)?(?:data|customers?|records?|rows?)|"
    r"(?:find|identify|discover|detect)\s+(?:natural\s+)?(?:groups?|segments?|clusters?)|"
    r"segment\s+(?:my\s+)?(?:customers?|data|users?|records?)|"
    r"group\s+(?:similar|my)\s+(?:customers?|records?|data)|"
    r"(?:k.?means|kmeans|k\s+means)|"
    r"natural\s+(?:groups?|segments?|clusters?)|"
    r"(?:what|which)\s+(?:groups?|segments?|clusters?)\s+(?:exist|are\s+there)|"
    r"(?:customer|data|record)\s+segmentation"
    r")\b",
    re.IGNORECASE,
)


def _detect_cluster_features(message: str, df: "pd.DataFrame") -> list[str] | None:
    """Extract specific feature columns to cluster on from the message.

    Returns a list of column names if specific numeric columns are mentioned,
    or None to use all numeric columns (auto-select).
    """
    msg_lower = message.lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    mentioned = [c for c in numeric_cols if c.lower() in msg_lower]
    return mentioned if len(mentioned) >= 2 else None


# Keywords that trigger a time-period comparison ("compare 2023 vs 2024")
_TIMEWINDOW_PATTERNS = re.compile(
    r"(?i)"
    r"compare\s+(?:\w+\s+)?(?:to|vs\.?|versus|and|against)\s+|"
    r"\b(20\d\d)\s+vs\.?\s+(20\d\d)\b|"
    r"\bQ[1-4](?:\s+20\d\d)?\s+vs\.?\s+Q[1-4]\b|"
    r"\b(year.over.year|yoy|year.on.year)\b|"
    r"\b(month.over.month|mom|month.on.month)\b|"
    r"\b(this year vs last year|last year vs this year)\b|"
    r"\b(this month vs last month|last month vs this month)\b|"
    r"\b(first half vs second half|H1 vs H2|h1 vs h2)\b|"
    r"\bhow.*(year|quarter|month|period).*(change|differ|compar)|"
    r"\b(period.comparison|time.window|compare.period|compare.date)",
    re.IGNORECASE,
)

# Sub-patterns for extracting specific date ranges
_YEAR_VS_PATTERN = re.compile(r"\b(20\d\d)\b.*?\b(20\d\d)\b", re.IGNORECASE)
_QUARTER_VS_PATTERN = re.compile(
    r"\bQ([1-4])(?:\s+(20\d\d))?\s+vs\.?\s*Q([1-4])(?:\s+(20\d\d))?\b", re.IGNORECASE
)
_HALF_PATTERN = re.compile(
    r"\b(first half|H1)\s+vs\.?\s*(second half|H2)\b", re.IGNORECASE
)
_YOY_PATTERN = re.compile(
    r"\b(year.over.year|yoy|year.on.year|this year vs last year)\b", re.IGNORECASE
)
_MOM_PATTERN = re.compile(
    r"\b(month.over.month|mom|month.on.month|this month vs last month)\b", re.IGNORECASE
)


def _detect_timewindow_request(message: str, df: "pd.DataFrame") -> dict | None:
    """Extract two time periods from the user message and the DataFrame.

    Returns a dict with keys:
      date_col, period1_name, period1_start, period1_end,
      period2_name, period2_start, period2_end
    or None if no date column or pattern can be resolved.
    """
    from core.analyzer import detect_time_columns

    date_cols = detect_time_columns(df)
    if not date_cols:
        return None
    date_col = date_cols[0]

    # Parse date column once
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    except Exception:  # noqa: BLE001
        return None
    if dates.empty:
        return None

    min_date = dates.min()
    max_date = dates.max()

    # --- Pattern 1: two explicit years ("compare 2023 vs 2024") ---
    m = _YEAR_VS_PATTERN.search(message)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 != y2:
            return {
                "date_col": date_col,
                "period1_name": str(y1),
                "period1_start": f"{y1}-01-01",
                "period1_end": f"{y1}-12-31",
                "period2_name": str(y2),
                "period2_start": f"{y2}-01-01",
                "period2_end": f"{y2}-12-31",
            }

    # --- Pattern 2: quarter vs quarter ("Q1 vs Q2", "Q3 2023 vs Q4 2023") ---
    m = _QUARTER_VS_PATTERN.search(message)
    if m:
        q1, opt_y1, q2, opt_y2 = (
            int(m.group(1)),
            m.group(2),
            int(m.group(3)),
            m.group(4),
        )
        # If no explicit year, use the most recent year in the data
        data_year = max_date.year
        y1 = int(opt_y1) if opt_y1 else data_year
        y2 = int(opt_y2) if opt_y2 else data_year
        _Q_STARTS = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        _Q_ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        return {
            "date_col": date_col,
            "period1_name": f"Q{q1} {y1}",
            "period1_start": f"{y1}-{_Q_STARTS[q1]}",
            "period1_end": f"{y1}-{_Q_ENDS[q1]}",
            "period2_name": f"Q{q2} {y2}",
            "period2_start": f"{y2}-{_Q_STARTS[q2]}",
            "period2_end": f"{y2}-{_Q_ENDS[q2]}",
        }

    # --- Pattern 3: "first half vs second half" / "H1 vs H2" ---
    if _HALF_PATTERN.search(message):
        data_year = max_date.year
        return {
            "date_col": date_col,
            "period1_name": f"H1 {data_year}",
            "period1_start": f"{data_year}-01-01",
            "period1_end": f"{data_year}-06-30",
            "period2_name": f"H2 {data_year}",
            "period2_start": f"{data_year}-07-01",
            "period2_end": f"{data_year}-12-31",
        }

    # --- Pattern 4: year-over-year ---
    if _YOY_PATTERN.search(message):
        latest_year = max_date.year
        prev_year = latest_year - 1
        return {
            "date_col": date_col,
            "period1_name": str(prev_year),
            "period1_start": f"{prev_year}-01-01",
            "period1_end": f"{prev_year}-12-31",
            "period2_name": str(latest_year),
            "period2_start": f"{latest_year}-01-01",
            "period2_end": f"{latest_year}-12-31",
        }

    # --- Pattern 5: month-over-month ---
    if _MOM_PATTERN.search(message):
        # Use last two complete months in the data
        latest_month_start = max_date.replace(day=1)
        prev_month_end = latest_month_start - pd.Timedelta(days=1)
        prev_month_start = prev_month_end.replace(day=1)
        # End of latest month
        import calendar

        latest_month_end = max_date.replace(
            day=calendar.monthrange(max_date.year, max_date.month)[1]
        )
        p1_label = prev_month_start.strftime("%b %Y")
        p2_label = latest_month_start.strftime("%b %Y")
        return {
            "date_col": date_col,
            "period1_name": p1_label,
            "period1_start": prev_month_start.strftime("%Y-%m-%d"),
            "period1_end": prev_month_end.strftime("%Y-%m-%d"),
            "period2_name": p2_label,
            "period2_start": latest_month_start.strftime("%Y-%m-%d"),
            "period2_end": latest_month_end.strftime("%Y-%m-%d"),
        }

    # --- Fallback: split the data date range in half ---
    midpoint = min_date + (max_date - min_date) / 2
    mid_str = midpoint.strftime("%Y-%m-%d")
    day_before_mid = (midpoint - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    p1_label = f"{min_date.strftime('%b %Y')} – {midpoint.strftime('%b %Y')}"
    p2_label = f"{midpoint.strftime('%b %Y')} – {max_date.strftime('%b %Y')}"
    return {
        "date_col": date_col,
        "period1_name": p1_label,
        "period1_start": min_date.strftime("%Y-%m-%d"),
        "period1_end": day_before_mid,
        "period2_name": p2_label,
        "period2_start": mid_str,
        "period2_end": max_date.strftime("%Y-%m-%d"),
    }


# Keywords that trigger a top-N / bottom-N ranking table
_TOPN_PATTERNS = re.compile(
    r"(?i)"
    r"\b(top|bottom|highest|lowest|best|worst|largest|smallest|most|fewest|least)\s+\d+\b|"
    r"\b(top|bottom|highest|lowest|best|worst|largest|smallest)\s+(five|ten|twenty|five|three)\b|"
    r"\bshow\s+me\s+(top|bottom|best|worst|highest|lowest)\b|"
    r"\brank(ed)?\s+(by|from)\b|"
    r"\bwho\s+are\s+(my\s+)?(best|worst|top|bottom|highest|lowest)\b|"
    r"\blist\s+(top|bottom|best|worst)\b|"
    r"\bwhich\s+(product|customer|region|category|item|store|order|account)s?\s+.{0,20}"
    r"(highest|lowest|most|fewest|best|worst)\b",
    re.IGNORECASE,
)

_TOPN_N_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twenty": 20,
    "fifty": 50,
}

_WHATIF_CHAT_PATTERNS = re.compile(
    r"(?i)"
    r"\bwhat\s+(if|would\s+happen\s+if|happens?\s+if)\b|"
    r"\bif\s+(i\s+)?(change[d]?|set|increase[d]?|decrease[d]?|double[d]?|halve[d]?|raise[d]?|lower[ed]?)\b|"
    r"\bsuppose\s+(that\s+)?\w+|"
    r"\bif\s+\w+\s+(was|were|is|equals?|becomes?|had|changed?\s+to)\b|"
    r"\bchange\s+\w+\s+to\b|"
    r"\bpredict\s+(with|for|where|if)\b|"
    r"\bwhat\s+would\s+(my\s+)?(prediction|result|forecast|output)\s+be\b|"
    r"\bhow\s+would\s+(the\s+)?prediction\s+change\b",
    re.IGNORECASE,
)


# Keywords that trigger a prediction error analysis ("where was my model wrong?")
# Note: no trailing \b — patterns use .* and words like "errors"/"mistakes" extend beyond the stem
_PRED_ERROR_PATTERNS = re.compile(
    r"\b(where.*model.*wrong|model.*wrong|wrong.*prediction|worst.*prediction|"
    r"prediction.*errors?|biggest.*errors?|largest.*errors?|"
    r"where.*fail|model.*fail|miss.*prediction|"
    r"which.*rows?.*wrong|which.*records?.*wrong|"
    r"show.*errors?|show.*mistakes?|prediction.*mistakes?)",
    re.IGNORECASE,
)

# Chat intent: show me the data / peek at rows / display records
# Intentionally excludes "show me top/bottom" (handled by TOPN) and
# "show errors/mistakes" (handled by PRED_ERROR above).
_RECORDS_PATTERNS = re.compile(
    r"(?i)"
    r"\bshow\s+me\s+(the\s+|my\s+)?(data|rows?|records?|table|dataset)\b|"
    r"\b(display|preview|peek\s+at|view)\s+(the\s+)?(data|rows?|records?|table|dataset)\b|"
    r"\blet\s+me\s+see\s+(the\s+)?(data|rows?|records?|dataset)\b|"
    r"\bwhat\s+does\s+(the\s+)?data\s+look\s+like\b|"
    r"\bshow\s+(first|last|next)\s+\d+\s+(rows?|records?|entries?|lines?)\b|"
    r"\bgive\s+me\s+(a\s+)?(sample|peek|look)\s+(of\s+)?(the\s+)?(data|rows?|records?)\b|"
    r"\bsample\s+(the\s+)?(data|rows?|records?)\b|"
    r"\bshow\s+rows?\s+where\b|"
    r"\bfind\s+(rows?|records?)\s+where\b|"
    r"\bshow\s+records?\s+where\b",
    re.IGNORECASE,
)


def _detect_records_request(message: str, df: "pd.DataFrame") -> dict:
    """Extract n, optional conditions, and offset from the user message.

    Returns dict with: n, conditions (list|None), offset
    """
    # Extract n from "first 20 rows", "show 10 records", etc.
    n = 20
    m_n = re.search(
        r"\b(first|last|next|show|top)?\s*(\d+)\s*(rows?|records?|entries?|lines?)\b",
        message,
        re.IGNORECASE,
    )
    if m_n:
        candidate = int(m_n.group(2))
        if 1 <= candidate <= 50:
            n = candidate

    # Extract optional WHERE clause: "where X op Y"
    conditions = None
    m_where = re.search(r"\bwhere\s+(.+)$", message, re.IGNORECASE)
    if m_where:
        where_clause = m_where.group(1).strip()
        from core.filter_view import parse_filter_request

        parsed = parse_filter_request(where_clause, list(df.columns))
        if parsed:
            conditions = parsed

    return {"n": n, "conditions": conditions, "offset": 0}


# Keywords that trigger a line/trend chart over time
_LINE_CHART_PATTERNS = re.compile(
    r"(?:"
    r"(?:plot|chart|graph|show|visualize)\s+(?:me\s+)?(?:the\s+)?(?:\w+\s+)?(?:over\s+time|by\s+(?:month|week|year|quarter|day|date))|"
    r"(?:trend|change)\s+(?:of|in|for)\s+|"
    r"\bline\s+chart\s+(?:of|for)\s+|"
    r"\bhow\s+(?:has|have|did)\s+\w+\s+(?:changed?|trended?|evolved?|moved?)|"
    r"\bshow\s+(?:me\s+)?(?:the\s+)?\w+\s+trend\b|"
    r"\btime\s+series\s+(?:of|for)\s+|"
    r"\b\w+\s+over\s+time\b|"
    r"\bplot\s+(?:the\s+)?trend\b|"
    r"\b(?:compare|overlay)\s+\w+\s+(?:and|vs\.?|versus|with)\s+\w+\b|"
    r"\b(?:compare|overlay)\s+\w+\s+and\s+\w+\s+(?:over\s+time|by\s+(?:month|week|year|quarter))\b"
    r")",
    re.IGNORECASE,
)


def _detect_line_chart_request(message: str, df: "pd.DataFrame") -> dict | None:
    """Extract value_cols and date_col for a line/trend chart request.

    Uses detect_time_columns() to find the date column automatically.
    Scans the message for ALL mentioned numeric column names (longest-match-first
    to avoid partial matches), enabling multi-column overlay charts.

    Returns dict with {value_cols: list[str], date_col: str} or None if no
    date column is detected.  value_cols always has at least one entry.
    """
    from core.analyzer import detect_time_columns as _dtc

    date_cols = _dtc(df)
    if not date_cols:
        return None

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None

    msg_lower = message.lower()
    # Collect ALL mentioned numeric columns (longest match first to prevent
    # shorter column names shadowing longer ones, e.g. "cost" inside "unit_cost")
    found: list[str] = []
    seen: set[str] = set()
    for col in sorted(numeric_cols, key=len, reverse=True):
        if col.lower() in msg_lower and col not in seen:
            found.append(col)
            seen.add(col)

    # Fall back to first numeric column when none are explicitly mentioned
    if not found:
        found = [numeric_cols[0]]

    return {"value_cols": found, "date_col": date_cols[0]}


# Keywords that trigger a box plot (distribution comparison, optionally by group)
_BOXPLOT_PATTERNS = re.compile(
    r"(?:"
    r"\bbox\s*(?:-\s*and\s*-\s*whisker|plot|chart)?\s*(?:of|for)\s+|"
    r"(?:distribution|spread|range|quartile|whisker)\s+(?:of|for)\s+\w+\s+by\s+|"
    r"(?:compare|show)\s+(?:the\s+)?(?:distribution|spread|range)\s+(?:of|for)\s+\w+\s+(?:by|across|per|for\s+each)\s+|"
    r"\boutliers?\s+(?:in|for)\s+\w+\s+by\s+|"
    r"\bwhisker\s+(?:plot|chart)\b"
    r")",
    re.IGNORECASE,
)


def _detect_boxplot_request(message: str, df: "pd.DataFrame") -> dict | None:
    """Extract value_col and optional group_col from a box plot request.

    Looks for a numeric value column and an optional categorical group column.
    Returns dict with {value_col, group_col} or None if no numeric column detected.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None

    categorical_cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() <= 30
        and df[c].nunique() >= 2
    ]

    msg_lower = message.lower()

    # Find value_col: numeric column mentioned in message (longest match first)
    value_col = None
    for col in sorted(numeric_cols, key=len, reverse=True):
        if col.lower() in msg_lower:
            value_col = col
            break
    if not value_col:
        value_col = numeric_cols[0]

    # Find group_col: categorical column mentioned after "by/across/per/for each"
    group_col = None
    by_match = re.search(
        r"\b(?:by|across|per|for\s+each)\s+([\w\s]{1,40}?)(?:\s*[.?!]|$)",
        msg_lower,
    )
    if by_match:
        fragment = by_match.group(1).strip()
        for col in sorted(categorical_cols, key=len, reverse=True):
            if col.lower() in fragment:
                group_col = col
                break

    # If no group_col found via "by" but a categorical column is mentioned
    if not group_col:
        for col in sorted(categorical_cols, key=len, reverse=True):
            if col.lower() in msg_lower and col != value_col:
                group_col = col
                break

    return {"value_col": value_col, "group_col": group_col}


# Keywords that trigger a pie / donut chart (composition / share breakdown)
_PIE_CHART_PATTERNS = re.compile(
    r"(?:"
    r"\bpie\s+(?:chart|graph|plot)\b|"
    r"\b(?:donut|doughnut)\s+(?:chart|graph|plot)\b|"
    r"\b(?:show|create|make|draw|generate|give\s+me|plot)\s+(?:me\s+)?(?:a\s+)?(?:pie|donut|doughnut)\b|"
    r"\b(?:composition|proportion|share|makeup)\s+(?:of\s+\w+\s+)?(?:chart|plot|pie|graph)\b|"
    r"\b(?:composition|proportion|share|makeup)\s+(?:by|of|for)\s+|"
    r"\bbreakdown\s+(?:chart|pie|plot)\b"
    r")",
    re.IGNORECASE,
)


def _detect_pie_chart_request(message: str, df: "pd.DataFrame") -> dict | None:
    """Extract value_col (numeric) and slice_col (categorical) for a pie chart.

    Looks for a numeric column to sum per slice and a categorical column to
    group by. Returns dict with {value_col, slice_col} or None if no usable
    columns found.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() >= 2
        and df[c].nunique() <= 30
    ]

    if not numeric_cols or not categorical_cols:
        return None

    msg_lower = message.lower()

    # Find slice_col: categorical column mentioned after "by/of/for/per/across"
    slice_col = None
    by_match = re.search(
        r"\b(?:by|of|for|per|across|grouped\s+by|segmented\s+by)\s+([\w\s]{1,40}?)(?:\s*[.?!]|$)",
        msg_lower,
    )
    if by_match:
        fragment = by_match.group(1).strip()
        for col in sorted(categorical_cols, key=len, reverse=True):
            if col.lower() in fragment:
                slice_col = col
                break

    # Fallback: first categorical column mentioned in message
    if not slice_col:
        for col in sorted(categorical_cols, key=len, reverse=True):
            if col.lower() in msg_lower:
                slice_col = col
                break

    # Final fallback: first categorical column in dataset
    if not slice_col:
        slice_col = categorical_cols[0]

    # Find value_col: numeric column mentioned in message (longest match first)
    value_col = None
    for col in sorted(numeric_cols, key=len, reverse=True):
        if col.lower() in msg_lower:
            value_col = col
            break

    # Fallback: first numeric column
    if not value_col:
        value_col = numeric_cols[0]

    return {"value_col": value_col, "slice_col": slice_col}


# ---------------------------------------------------------------------------
# Bar / column chart via chat
# ---------------------------------------------------------------------------

_BAR_CHART_PATTERNS = re.compile(
    r"(?:"
    r"\bbar\s+(?:chart|graph|plot)\b|"
    r"\bcolumn\s+(?:chart|graph|plot)\b|"
    r"\bvertical\s+bar\b|"
    r"\b(?:show|create|make|draw|generate|give\s+me|plot)\s+(?:me\s+)?(?:a\s+)?bar\s+(?:chart|graph)\b|"
    r"\b(?:bar|column)\s+(?:chart|graph)\s+(?:of|for|showing)\b"
    r")",
    re.IGNORECASE,
)


def _detect_bar_chart_request(message: str, df: "pd.DataFrame") -> dict | None:
    """Extract value_col (numeric) and optional group_col (categorical) for a bar chart.

    Looks for 'by/per/for each <group_col>' clause for the group, and the longest
    matching numeric column name in the message for the value.

    Returns dict with {value_col, group_col, agg} or None if no numeric column found.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() >= 2
        and df[c].nunique() <= 50
    ]

    if not numeric_cols:
        return None

    msg_lower = message.lower()

    # Find value_col: numeric column mentioned in message (longest match first)
    value_col = None
    for col in sorted(numeric_cols, key=len, reverse=True):
        if col.lower() in msg_lower or col.replace("_", " ").lower() in msg_lower:
            value_col = col
            break
    if not value_col:
        value_col = numeric_cols[0]

    # Find group_col via "by/per/for each" clause
    group_col = None
    if categorical_cols:
        by_match = re.search(
            r"\b(?:by|per|for\s+each|grouped\s+by|grouped\s+on)\s+([\w\s]{1,40}?)(?:\s*[.?!]|$)",
            msg_lower,
        )
        if by_match:
            fragment = by_match.group(1).strip()
            for col in sorted(categorical_cols, key=len, reverse=True):
                if col.lower() in fragment or col.replace("_", " ").lower() in fragment:
                    group_col = col
                    break

        # Fallback: categorical column mentioned anywhere in message
        if not group_col:
            for col in sorted(categorical_cols, key=len, reverse=True):
                if (
                    col.lower() in msg_lower
                    or col.replace("_", " ").lower() in msg_lower
                ):
                    group_col = col
                    break

        # Final fallback: first categorical column
        if not group_col:
            group_col = categorical_cols[0]

    # Detect aggregation keyword
    agg = "sum"
    if re.search(r"\b(?:average|mean|avg)\b", msg_lower):
        agg = "mean"
    elif re.search(r"\bcount\b", msg_lower):
        agg = "count"
    elif re.search(r"\bmax(?:imum)?\b", msg_lower):
        agg = "max"
    elif re.search(r"\bmin(?:imum)?\b", msg_lower):
        agg = "min"

    return {"value_col": value_col, "group_col": group_col, "agg": agg}


# ---------------------------------------------------------------------------
# Dataset download / export via chat
# ---------------------------------------------------------------------------

_DOWNLOAD_PATTERNS = re.compile(
    r"(?:"
    r"\bdownload\s+(?:my\s+)?(?:the\s+)?(?:data(?:set)?|csv|file|filtered\s+data)\b|"
    r"\bexport\s+(?:my\s+)?(?:the\s+)?(?:data(?:set)?|csv|results?|records?)\b|"
    r"\bsave\s+(?:the\s+)?(?:data(?:set)?|csv)\s*(?:to|as)?\s*(?:csv|file)?\b|"
    r"\b(?:give\s+me)\s+(?:the\s+)?(?:data(?:set)?\s+(?:as\s+)?csv|csv\s+(?:export|file|download))\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Histogram / frequency distribution via chat
# ---------------------------------------------------------------------------

_HISTOGRAM_PATTERNS = re.compile(
    r"(?:"
    r"\bhistogram\s+(?:of|for)\b|"
    r"\b(?:show|create|make|draw|generate|give\s+me|plot)\s+(?:me\s+)?(?:a\s+)?histogram\b|"
    r"\bfrequency\s+histogram\s+(?:of|for)\b|"
    r"\bbinned?\s+distribution\s+(?:of|for)\b|"
    r"\bfrequency\s+chart\s+(?:of|for)\b|"
    r"\bdistribution\s+(?:chart|histogram)\s+(?:of|for)\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Missing values / null overview via chat
# ---------------------------------------------------------------------------

_NULL_MAP_PATTERNS = re.compile(
    r"(?:"
    r"\b(?:show|display|list|tell\s+me|give\s+me)\s+(?:me\s+)?(?:the\s+)?missing\s+(?:values?|data|fields?)\b|"
    r"\bwhich\s+columns?\s+(?:have|has|contain)\s+(?:missing|null|blank|empty)\b|"
    r"\b(?:null|missing)\s+(?:values?|data)\s+(?:overview|summary|report|map|by\s+column)\b|"
    r"\bdata\s+completeness\s+(?:overview|summary|by\s+column|per\s+column|breakdown)\b|"
    r"\bhow\s+(?:many|much)\s+missing\s+(?:values?|data)\b|"
    r"\b(?:null|missing)\s+(?:count|rate|percentage|percent)\s+(?:by|per|for\s+each)\s+column\b|"
    r"\bwhere\s+(?:is|are)\s+(?:my\s+)?missing\s+(?:data|values?)\b"
    r")",
    re.IGNORECASE,
)


# Keywords that trigger a full-dataset summary statistics table
_SUMMARY_STATS_PATTERNS = re.compile(
    r"(?:"
    r"\b(?:summarize|describe|overview\s+of)\s+(?:all\s+)?(?:my\s+)?(?:data|dataset|all\s+columns?)\b|"
    r"\bstatistical\s+(?:summary|overview)\b|"
    r"\bsummary\s+statistics\b|"
    r"\bdescriptive\s+statistics\b|"
    r"\bstats\s+(?:for\s+)?(?:all\s+)?(?:my\s+)?(?:columns?|data|dataset)\b|"
    r"\b(?:data|dataset)\s+(?:statistics|stats|summary|overview)\b|"
    r"\b(?:show|give)\s+me\s+(?:all\s+)?(?:the\s+)?(?:statistics|stats|summary)\s+"
    r"(?:for\s+)?(?:all\s+)?(?:my\s+)?(?:data|columns?)\b"
    r")",
    re.IGNORECASE,
)

# Keywords that trigger a single-column value frequency table
_VALUE_COUNT_PATTERNS = re.compile(
    r"(?i)(?:"
    r"\bmost\s+(?:common|frequent)\s+(?:values?\s+(?:in|for|of)\s+\w+|\w+\s+values?|\w+)\b|"
    r"\bfrequency\s+(?:table\s+)?(?:for|of)\s+\w+\b|"
    r"\bvalue\s+(?:counts?|frequencies?)\s+(?:for|of)\s+\w+\b|"
    r"\bhow\s+(?:often|common|frequent)\s+(?:does|do|is|are)\s+(?:each\s+)?\w+\s*(?:value\s+)?(?:appear|occur|show\s+up)\b|"
    r"\bhow\s+(?:common|frequent)\s+(?:is|are)\s+(?:each\s+)?\w+\b|"
    r"\bcount\s+(?:of\s+)?(?:each|every)\s+\w+\s+(?:value|occurrence)\b|"
    r"\bhow\s+is\s+(?:my\s+)?data\s+split\s+(?:by|across)\s+\w+\b|"
    r"\btop\s+(?:values?\s+(?:in|for|of)|occurrences?\s+(?:in|for))\s+\w+\b"
    r")",
    re.IGNORECASE,
)


def _detect_value_counts_col(message: str, df: "pd.DataFrame") -> str | None:
    """Extract the categorical column for value counts from the user message.

    Scans actual DataFrame column names (longest first) against the message.
    Falls back to the first categorical column if no column name is found.
    Returns the column name or None if the DataFrame has no columns.
    """
    if df.empty or len(df.columns) == 0:
        return None
    msg_lower = message.lower()
    # Try longest match first to avoid partial matches
    for col in sorted(df.columns, key=len, reverse=True):
        if col.lower() in msg_lower or col.replace("_", " ").lower() in msg_lower:
            return col
    # Fallback: first non-numeric column, else first column
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        return cat_cols[0]
    return df.columns[0]


# Keywords that trigger pair correlation analysis between two specific columns
# Does NOT overlap with _CORRELATION_TARGET_PATTERNS (which needs a single target) or
# _HEATMAP_PATTERNS (which requires "matrix/heatmap/pairwise/all columns")
_PAIR_CORR_PATTERNS = re.compile(
    r"(?:"
    r"\bcorrelation\s+between\s+\w[\w\s]*\s+and\s+\w|"
    r"\bhow\s+(?:strongly\s+)?correlated\s+(?:are|is)\s+\w[\w\s]*\s+(?:and|with|to)\s+\w|"
    r"\bpearson\s+(?:r\b|correlation|coefficient)|"
    r"\br\s*(?:value|coefficient)\s+(?:for|between|of)\s+\w|"
    r"\bdoes\s+\w[\w\s]*\s+correlate\s+with\s+\w|"
    r"\bcorrelation\s+of\s+\w[\w\s]*\s+(?:with|vs|versus|and|against)\s+\w|"
    r"\bhow\s+(?:closely|strongly)?\s+(?:related|linked)\s+(?:are|is)\s+\w[\w\s]*\s+(?:and|to|with)\s+\w"
    r")",
    re.IGNORECASE,
)


def _detect_pair_corr_cols(message: str, df: "pd.DataFrame") -> tuple[str, str] | None:
    """Extract the two column names for a pair correlation request.

    Scans actual DataFrame column names (longest first) to find two mentions.
    Returns (col1, col2) or None if fewer than 2 numeric columns found.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return None

    msg_lower = message.lower()
    found: list[str] = []
    for col in sorted(numeric_cols, key=len, reverse=True):
        if col.lower() in msg_lower or col.replace("_", " ").lower() in msg_lower:
            if col not in found:
                found.append(col)
        if len(found) == 2:
            break

    if len(found) == 2:
        return found[0], found[1]
    # Fallback: first two numeric columns
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]
    return None


# Keywords that trigger a single-column aggregate statistic query
# ("what's the average of revenue?", "total sales", "max cost", "count rows")
# Does NOT overlap with _GROUP_PATTERNS (no "by" clause) or _COLUMN_PROFILE_PATTERNS
# (no "describe/tell me about"). Requires explicit aggregation word.
_STAT_QUERY_PATTERNS = re.compile(
    r"(?:"
    r"\bwhat(?:'s|\s+is)?\s+the\s+(?:average|mean|median|total|sum|max(?:imum)?|min(?:imum)?|count|std(?:ev(?:iation)?)?)\s+(?:of|for|value\s+of)?\s*\w|"
    r"\b(?:average|mean|median|sum|total)\s+(?:of|for)\s+\w|"
    r"\b(?:max(?:imum)?|min(?:imum)?)\s+(?:value\s+(?:of|for)\s+\w|\w+\s+value)|"
    r"\bhow\s+(?:many|much)\s+(?:total|rows?|records?|entries?)\b|"
    r"\bcount\s+(?:the\s+)?(?:rows?|records?|entries?|total)\b|"
    r"\btotal\s+(?:number\s+of|count\s+of)?\s*\w+\s+(?:is|are|=)?\b|"
    r"\b(?:sum|total)\s+\w+\b"
    r")",
    re.IGNORECASE,
)

# Map natural-language aggregation words to canonical agg names
_AGG_WORD_MAP = {
    "average": "mean",
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "total": "sum",
    "maximum": "max",
    "max": "max",
    "minimum": "min",
    "min": "min",
    "count": "count",
    "std": "std",
    "stdev": "std",
    "stddev": "std",
    "standard deviation": "std",
}


def _detect_stat_query(message: str, df: "pd.DataFrame") -> dict | None:
    """Extract agg and col from a stat query message.

    Returns {"agg": ..., "col": ...} or None if no aggregation detected.
    """
    msg_lower = message.lower()

    # Detect aggregation word — check count/how-many FIRST to avoid
    # "total rows" being captured as "sum" via the "total" word mapping
    agg: str | None = None
    if "how many" in msg_lower or re.search(
        r"\bcount\s+(?:the\s+)?(?:rows?|records?|entries?|total)\b", msg_lower
    ):
        agg = "count"
    else:
        for phrase, canonical in sorted(
            _AGG_WORD_MAP.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if phrase in msg_lower:
                agg = canonical
                break
    if not agg:
        return None

    # Detect column name (longest-match first)
    col: str | None = None
    for c in sorted(df.columns, key=len, reverse=True):
        if c.lower() in msg_lower or c.replace("_", " ").lower() in msg_lower:
            col = c
            break

    # For count, col is optional
    if agg == "count":
        return {"agg": "count", "col": col}

    # For other aggregations, need a numeric col
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not col or col not in numeric_cols:
        # Fallback to first numeric column
        if numeric_cols:
            col = numeric_cols[0]
        else:
            return None

    return {"agg": agg, "col": col}


_GROUP_TREND_PATTERNS = re.compile(
    r"(?:"
    r"which\s+\w+\s+(?:are|is)\s+(?:growing|trending|increasing|declining|falling|rising)\b|"
    r"fastest\s+(?:growing|rising|declining|falling)\s+\w+|"
    r"(?:growth|trend|trending)\s+(?:rate\s+)?(?:by|per|for\s+each)\s+\w+|"
    r"which\s+\w+\s+(?:has|have)\s+(?:the\s+)?(?:most\s+)?(?:growth|increase|decline|decrease)\b|"
    r"how\s+(?:are|is)\s+(?:my\s+)?\w+\s+(?:trending|growing|changing\s+over\s+time)\b|"
    r"trending\s+(?:up|down)\s+(?:by|per|for\s+each)\s+\w+|"
    r"compare\s+(?:growth|trend)s?\s+(?:across|by|per|for)\s+\w+"
    r")",
    re.IGNORECASE,
)


# Keywords that trigger conversation export / analysis report download
_CONV_EXPORT_PATTERNS = re.compile(
    r"\b(?:export|download|save|share|send)\b.*\b(?:conversation|analysis|report|summary|chat|transcript|journey|story)\b"
    r"|\bsave this analysis\b|\bshare this analysis\b|\bshare this report\b"
    r"|\bexport this\b|\bdownload this\b"
    r"|\bgenerate a report\b|\bcreate a report\b|\bmake a report\b"
    r"|\bshare my findings\b|\bexport my findings\b"
    r"|\bdownload the chat\b|\bshare the conversation\b|\bexport the conversation\b",
    re.IGNORECASE,
)


# Keywords that trigger the auto-retrain status/toggle card
_AUTO_RETRAIN_PATTERNS = re.compile(
    r"\b(auto.?retrain|automatic.*retrain|retrain.*automatic|"
    r"retrain.*new.*data|new.*data.*retrain|retrain.*upload|upload.*retrain|"
    r"retrain.*schedul|schedul.*retrain|keep.*model.*fresh|keep.*model.*current|"
    r"auto.*train.*new|train.*new.*data.*auto|"
    r"enable.*auto|disable.*auto|turn.*on.*retrain|turn.*off.*retrain|"
    r"auto.?retrain.*status|status.*auto.?retrain|"
    r"retrain.*when.*upload|automatically.*when.*upload)\b",
    re.IGNORECASE,
)

_PREDICT_OPP_PATTERNS = re.compile(
    r"(?i)\b("
    r"what\s+can\s+I\s+predict\b|"
    r"what\s+should\s+I\s+(?:model|predict)\b|"
    r"suggest\s+(?:a\s+)?(?:prediction\s+)?target\b|"
    r"what\s+(?:can\s+I\s+|would\s+be\s+good\s+to\s+|is\s+worth\s+)?predict(?:ing)?\b|"
    r"help\s+me\s+choose\s+(?:a\s+)?(?:prediction\s+)?target\b|"
    r"what\s+(?:columns?|variables?)\s+(?:can|should)\s+I\s+(?:predict|model)\b|"
    r"what\s+(?:can\s+this\s+data|can\s+my\s+data|is\s+good\s+to)\s+predict\b|"
    r"prediction\s+opportunities?\b|"
    r"what\s+(?:models?|predictions?)\s+(?:are\s+)?(?:possible|worth\s+building)\b"
    r")",
    re.IGNORECASE,
)

_DATASET_COMPARE_PATTERNS = re.compile(
    r"(?i)("
    r"(?:what\s+)?(?:changed|different|change[ds]?)\s+(?:in|with|about)\s+(?:my\s+)?(?:new\s+)?data|"
    r"how\s+(?:does|is)\s+(?:my\s+)?(?:new\s+data|new\s+dataset|this\s+data)\s+(?:compare|different|look)|"
    r"(?:compare|comparison)\s+(?:the\s+)?(?:datasets?|data\s+files?|uploads?)|"
    r"distribution\s+(?:shift|change|drift|comparison)|"
    r"(?:what|any)\s+(?:distribution|data)\s+changes|"
    r"(?:has|have)\s+(?:the\s+)?(?:my\s+)?data\s+(?:changed|shifted|drifted)|"
    r"new\s+vs\.?\s+old\s+data|"
    r"(?:differences?|changes?)\s+between\s+(?:my\s+)?(?:datasets?|uploads?|files?)|"
    r"is\s+my\s+new\s+data\s+(?:different|similar|compatible)"
    r")",
    re.IGNORECASE,
)

_VERSION_HISTORY_PATTERNS = re.compile(
    r"(?i)("
    r"(?:show|view|see|display)\s+(?:my\s+)?(?:data\s+)?(?:version|upload|dataset)\s+(?:history|timeline|log|versions)\b|"
    r"(?:how\s+(?:many|much)|list|what)\s+(?:versions?|uploads?|datasets?)\s+(?:do\s+I\s+have|have\s+I\s+uploaded|are\s+there)\b|"
    r"(?:data|dataset)\s+version\s+history\b|"
    r"(?:upload|data)\s+(?:history|timeline|log)\b|"
    r"(?:history|timeline)\s+of\s+(?:my\s+)?(?:uploads?|datasets?)\b|"
    r"(?:how\s+(?:has|have)\s+(?:my\s+)?data\s+(?:evolved|changed|progressed)\s+over\s+(?:time|versions?))\b|"
    r"(?:track|show)\s+(?:my\s+)?(?:data\s+)?changes\s+over\s+(?:time|versions?)\b|"
    r"(?:all\s+)?(?:my\s+)?(?:previous|past)\s+(?:uploads?|datasets?|versions?)\b"
    r")",
    re.IGNORECASE,
)

_HEALTH_SUMMARY_PATTERNS = re.compile(
    r"\b("
    r"how\s+(?:are|is)\s+(?:my\s+)?(?:model|models|deployment|deployments)\s+(?:doing|performing|holding up)\b|"
    r"(?:any\s+)?(?:issues?|problems?|alerts?)\s+(?:with\s+)?(?:my\s+)?(?:model|models|deployment|deployments)\b|"
    r"(?:model|deployment)\s+(?:health|status|check)\b|"
    r"check\s+(?:my\s+)?(?:model|models|deployment|deployments)\b|"
    r"(?:are|is)\s+(?:my\s+)?(?:model|models|deployment|deployments)\s+(?:ok|okay|healthy|still good|up to date|still accurate|still working)\b|"
    r"(?:model|deployment|prediction)\s+drift\b|"
    r"(?:are|is)\s+(?:my\s+)?(?:model|prediction)\s+(?:still\s+)?(?:accurate|current|fresh|working)\b|"
    r"(?:stale|outdated|old)\s+(?:model|models|deployment)\b|"
    r"(?:do\s+I\s+need\s+to\s+retrain|should\s+I\s+retrain|time\s+to\s+retrain)\b"
    r")",
    re.IGNORECASE,
)

_FEATURE_SEL_PATTERNS = re.compile(
    r"(?:"
    r"(?:are\s+)?(?:all|my)\s+(?:columns?|features?)\s+(?:useful|important|helpful|contributing|relevant)\b|"
    r"(?:which|what)\s+(?:columns?|features?)\s+(?:(?:are|is)\s+)?(?:not\s+)?(?:useful|important|helpful|needed|contributing)\b|"
    r"(?:remove|drop|exclude|eliminate)\s+(?:unimportant|weak|useless|low.importance|low.value|irrelevant)\s+(?:columns?|features?)\b|"
    r"feature\s+selection\b|"
    r"(?:identify|find)\s+(?:weak|useless|unimportant|low.importance)\s+(?:columns?|features?)\b|"
    r"(?:reduce|trim|prune)\s+(?:my\s+)?(?:features?|columns?)\b|"
    r"(?:which|what)\s+(?:columns?|features?)\s+should\s+I\s+(?:remove|drop|exclude|keep)\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Inline multi-feature prediction via chat (distinct from what-if which changes
# ONE feature; this accepts MULTIPLE explicit feature values from the message)
# ---------------------------------------------------------------------------
_INLINE_PRED_PATTERNS = re.compile(
    r"(?i)(?:"
    r"(?:run|make|give\s+me|calculate|compute|get)\s+a?\s*prediction\s+(?:for|with|where|using|given)\b|"
    r"predict\s+(?:for\s+(?:me\s+)?)?(?:these|the\s+following|my)\s+(?:values?|inputs?|numbers?|data|scenario)\b|"
    r"(?:what|estimate)\s+(?:would|will|is)\s+(?:my\s+)?(?:\w+\s+)?(?:be|equal|come\s+to)\s+(?:if|for|with|when|given)\b|"
    r"(?:score|classify|evaluate)\s+(?:this|these|my)\s+(?:record|example|instance|scenario|case|data)\b|"
    r"(?:run|apply|use)\s+(?:the\s+)?model\s+(?:on|with|for|to|given)\b|"
    r"(?:plug|put|input|enter)\s+(?:these|the\s+following|these\s+values?)\s+into\s+the\s+model\b|"
    r"model\s+output\s+(?:for|with|given|when)\b|"
    r"what\s+does\s+(?:the\s+)?model\s+(?:say|predict|give|output)\s+(?:for|with|when|if|given)\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Multi-row batch prediction patterns:
# "predict for: Region=East, Units=100; Region=West, Units=150"
# "run predictions for these scenarios: scenario1; scenario2"
# "batch predict: X=1; X=2; X=3"
# Key differentiator: presence of ";" separator between rows
# ---------------------------------------------------------------------------
_MULTI_ROW_PRED_PATTERNS = re.compile(
    r"(?i)(?:"
    r"(?:predict|run\s+predictions?|make\s+predictions?|score|get\s+predictions?)\s+"
    r"(?:for\s+(?:these|multiple|several|the\s+following)\b|"
    r"(?:these|multiple|several)\s+(?:scenarios?|records?|cases?|inputs?|rows?)\b)|"
    r"(?:batch|bulk|multiple)\s+(?:predict(?:ion)?s?|scenarios?)\b|"
    r"predictions?\s+for\s+(?:each|all\s+of\s+)?(?:these|multiple|several)\b|"
    r"compare\s+(?:these|multiple|several)\s+(?:scenarios?|inputs?|predictions?)\b|"
    r"run\s+(?:the\s+)?model\s+(?:on|for)\s+(?:multiple|several|these)\b"
    r")",
    re.IGNORECASE,
)


def _extract_multi_row_predictions(
    message: str, feature_names: list[str]
) -> list[dict[str, object]]:
    """Parse multiple prediction rows separated by semicolons.

    Each segment is parsed by the existing _extract_multi_feature_prediction helper.
    Returns a list of feature dicts only when 2+ valid rows are found.

    Leading text (e.g., "predict for:") before the first key=value pair is stripped
    from each segment to avoid false key-value matches like "for: Region" → key="for".
    """
    # Build a regex that finds the start of the first known feature key in a segment
    # so we can strip any leading non-kv preamble ("predict for:", "scenario 1:", etc.)
    _name_lower = {f.lower() for f in feature_names}

    def _trim_preamble(segment: str) -> str:
        """Return segment starting from the first occurrence of a known feature=value."""
        import re as _re

        for match in _re.finditer(r"\b([A-Za-z_][\w\s]{0,20}?)\s*=", segment):
            key_candidate = match.group(1).strip().lower()
            if (
                key_candidate in _name_lower
                or key_candidate.replace(" ", "_") in _name_lower
            ):
                return segment[match.start() :]
        return segment

    # Split by semicolons — the analyst-natural separator for multiple scenarios
    segments = message.split(";")
    rows: list[dict[str, object]] = []
    for segment in segments:
        trimmed = _trim_preamble(segment.strip())
        extracted = _extract_multi_feature_prediction(trimmed, feature_names)
        if extracted:
            rows.append(extracted)
    # Require at least 2 rows to distinguish from single inline prediction
    return rows if len(rows) >= 2 else []


# Keywords that trigger a sensitivity / sweep analysis:
# "how sensitive is revenue to units", "sensitivity analysis on price",
# "sweep price from 10 to 100", "how does prediction change as units varies"
# Keywords that trigger the guided onboarding wizard card
_ONBOARDING_PATTERNS = re.compile(
    r"(?:"
    r"(?:guide|help|walk)\s+me\s+(?:through|along|step|get\s+started)\b|"
    r"(?:get|getting)\s+started\b|"
    r"(?:how\s+do\s+I|where\s+do\s+I)\s+(?:start|begin|use\s+this|get\s+started)\b|"
    r"(?:show|give)\s+me\s+(?:the\s+)?(?:steps?|guide|tutorial|walkthrough|wizard|onboard)\b|"
    r"\bonboarding\b|"
    r"what\s+(?:should|do)\s+I\s+do\s+(?:first|next|now)\b|"
    r"(?:first|next)\s+steps?\b|"
    r"(?:new|first.time)\s+(?:user|analyst|here)\b"
    r")",
    re.IGNORECASE,
)


_SENSITIVITY_PATTERNS = re.compile(
    r"(?i)"
    r"sensitivity\s+(?:analysis\s+(?:on|for|of)\s+|(?:of|for)\s+)?\w|"
    r"how\s+sensitive\s+is\b|"
    r"(?:sweep|vary|range)\s+\w+\s+from\b|"
    r"how\s+does\s+(?:the\s+)?(?:prediction|model|output|result|forecast)\s+change\s+as\b|"
    r"effect\s+of\s+\w+\s+on\s+(?:the\s+)?(?:prediction|model|output|result)\b|"
    r"(?:show|plot|chart)\s+(?:me\s+)?(?:how|the\s+effect\s+of)\s+\w+\s+(?:affects?|changes?|impacts?)\b|"
    r"what\s+happens?\s+(?:to\s+)?(?:the\s+)?(?:prediction|result|output)\s+as\s+\w+\s+(?:varies?|increases?|decreases?)\b|"
    r"run\s+a\s+sensitivity\b",
    re.IGNORECASE,
)


_LEARNING_CURVE_PATTERNS = re.compile(
    r"(?i)(?:"
    r"(?:would|will|does?)\s+(?:more|additional|extra)\s+(?:data|rows?|training)\s+(?:help|improve|boost)\b|"
    r"learning\s+curve\b|"
    r"how\s+(?:much|many)\s+(?:more\s+)?(?:data|rows?)\s+(?:do\s+I\s+need|should\s+I\s+(?:get|collect|add))\b|"
    r"(?:do|does?|did)\s+(?:I|my\s+model)\s+(?:need|have enough)\s+(?:data|rows?)\b|"
    r"(?:is|are)\s+my\s+(?:training\s+)?(?:data|dataset)\s+(?:big\s+enough|sufficient|enough)\b|"
    r"(?:would|will)\s+(?:collecting|adding|getting|having|gathering)\s+more\s+(?:data|rows?)\b|"
    r"model.*(?:converge|converged|plateau|saturate)\b|"
    r"(?:data|training)\s+size\s+(?:analysis|curve|impact|effect)\b"
    r")",
    re.IGNORECASE,
)

# Partial Dependence Plot — marginal effect of one feature averaged over training data.
# Distinct from sensitivity analysis (fixes others at means) — PDP averages over the
# actual training distribution. Business analysts ask:
#   "partial dependence for price", "marginal effect of units on revenue",
#   "how does price affect predictions on average", "PDP for region",
#   "average effect of units on my model", "population-level effect of discount"
_PDP_PATTERNS = re.compile(
    r"(?i)(?:"
    r"partial\s+depend(?:ence|ency)?\s+(?:plot\s+)?(?:for|of|on)\s+\w|"
    r"pdp\s+(?:for|of|on)\s+\w|"
    r"marginal\s+effect\s+of\s+\w|"
    r"(?:how\s+does|what\s+is\s+the\s+effect\s+of)\s+\w+\s+(?:affect|influence|impact)\s+(?:the\s+)?prediction[s]?\s+on\s+average|"
    r"average\s+effect\s+of\s+\w+\s+on\s+(?:the\s+)?(?:prediction|model|output)|"
    r"population.level\s+effect\s+of\s+\w|"
    r"how\s+does\s+\w+\s+(?:relate\s+to|drive|affect)\s+(?:the\s+)?(?:average|mean)\s+prediction|"
    r"partial\s+dependence\b"
    r")",
    re.IGNORECASE,
)


def _detect_pdp_feature(message: str, feature_names: list[str]) -> str | None:
    """Extract the feature name to sweep from a PDP request.

    Scans the message for the longest-match column name from the model's feature list.
    Returns None when no feature name is recognised.
    """
    msg_lower = message.lower()
    # Sort by length descending so "product_category" beats "category"
    for feat in sorted(feature_names, key=len, reverse=True):
        if feat.lower() in msg_lower or feat.lower().replace("_", " ") in msg_lower:
            return feat
    return None


# Calibration check — how reliable are the model's confidence scores?
_CALIBRATION_CHECK_PATTERNS = re.compile(
    r"(?i)(?:"
    r"(?:how\s+)?(?:well.)?calibrated\s+(?:is\s+)?(?:(?:the|my)\s+)?(?:model|classifier|predictions?)\b|"
    r"(?:are|is)\s+(?:my\s+)?(?:model.s\s+)?confidence\s+scores?\s+(?:reliable|accurate|trustworthy|calibrated)\b|"
    r"reliability\s+diagram\b|"
    r"(?:show|check|plot|display|see|view)\s+(?:me\s+)?(?:the\s+)?calibration\b|"
    r"brier\s+score\b|"
    r"(?:model\s+)?calibration\s+(?:check|curve|plot|analysis|report)\b|"
    r"(?:how\s+accurate\s+are|can\s+I\s+trust)\s+(?:the\s+)?(?:confidence|probability|prob)\s+(?:scores?|estimates?)\b"
    r")",
    re.IGNORECASE,
)


# Feature interaction — 2-D heatmap sweeping two features jointly
_INTERACTION_PATTERNS = re.compile(
    r"(?i)(?:"
    r"interaction\s+(?:between|of)\s+\w+\s+and\s+\w+\b|"
    r"how\s+do\s+\w+\s+and\s+\w+\s+(?:interact|together|jointly|combine)\b|"
    r"(?:joint|combined|dual|2d|two.dimensional)\s+(?:effect|sensitivity|analysis|heatmap|plot)\b|"
    r"(?:show|plot|chart)\s+(?:me\s+)?(?:the\s+)?interaction\s+(?:between|of)\b|"
    r"how\s+do\s+(?:both\s+)?\w+\s+and\s+\w+\s+(?:affect|impact|influence|change)\s+(?:the\s+)?(?:prediction|model|output|result)\b|"
    r"effect\s+of\s+\w+\s+and\s+\w+\s+(?:together|jointly|combined)\b|"
    r"(?:feature\s+)?interaction\s+(?:plot|heatmap|map|analysis|grid)\b|"
    r"(?:2d|two.way)\s+sensitivity\b"
    r")",
    re.IGNORECASE,
)


def _detect_interaction_request(
    message: str,
    feature_names: list[str],
) -> dict | None:
    """Extract two feature names from an interaction request.

    Scans for the two longest column names that appear in the message.
    Returns {"feature1": ..., "feature2": ...} or None if fewer than 2 found.
    """
    msg_lower = message.lower().replace("-", "_").replace(" ", "_")
    # Sort longest-first so "unit_cost" is matched before "unit"
    sorted_feats = sorted(feature_names, key=len, reverse=True)
    found: list[str] = []
    for feat in sorted_feats:
        if feat.lower() in msg_lower and feat not in found:
            found.append(feat)
        if len(found) == 2:
            break
    if len(found) < 2:
        return None
    return {"feature1": found[0], "feature2": found[1]}


# ---------------------------------------------------------------------------
# Analysis Template patterns
# ---------------------------------------------------------------------------

_SAVE_TEMPLATE_PATTERNS = re.compile(
    r"(?i)(?:"
    r"save\s+(?:this\s+)?(?:analysis|queries|questions|session|conversation)\s+as\s+(?:a\s+)?(?:template|script)\b|"
    r"create\s+(?:a\s+)?(?:analysis\s+)?template\s+(?:called|named|for)\b|"
    r"(?:bookmark|save)\s+(?:these\s+)?(?:queries|questions|steps)\s+as\b|"
    r"save\s+(?:as|this\s+as)\s+(?:a\s+)?template\b|"
    r"make\s+(?:this\s+)?(?:a\s+)?(?:reusable\s+)?template\b|"
    r"save\s+(?:my|this)\s+(?:analysis\s+)?(?:workflow|flow|sequence)\b"
    r")",
    re.IGNORECASE,
)

_LIST_TEMPLATES_PATTERNS = re.compile(
    r"(?i)(?:"
    r"(?:show|list|what\s+are|see)\s+(?:my\s+)?(?:saved\s+)?(?:analysis\s+)?templates?\b|"
    r"(?:do\s+I\s+have|have\s+I\s+saved)\s+(?:any\s+)?(?:analysis\s+)?templates?\b|"
    r"(?:my\s+)?(?:analysis\s+)?templates?\s+(?:list|saved)\b|"
    r"what\s+templates?\s+(?:do\s+I\s+have|have\s+I\s+saved)\b"
    r")",
    re.IGNORECASE,
)

_REPLAY_TEMPLATE_PATTERNS = re.compile(
    r"(?i)(?:"
    r"replay\s+(?:my\s+)?(?:the\s+)?['\"]?[\w\s]+['\"]?\s+template\b|"
    r"run\s+(?:my\s+)?(?:the\s+)?['\"]?[\w\s]+['\"]?\s+(?:template|analysis\s+again)\b|"
    r"apply\s+(?:my\s+)?(?:the\s+)?['\"]?[\w\s]+['\"]?\s+template\b|"
    r"use\s+(?:my\s+)?(?:the\s+)?['\"]?[\w\s]+['\"]?\s+template\b|"
    r"re.?run\s+(?:my\s+)?(?:the\s+)?['\"]?[\w\s]+['\"]?\s+(?:template|analysis)\b|"
    r"replay\s+(?:my\s+)?(?:last|saved|previous)\s+(?:analysis|queries|questions|template)\b"
    r")",
    re.IGNORECASE,
)

# Extract template name from save/replay messages
_TEMPLATE_NAME_RE = re.compile(
    r"""(?:called|named|as|for)\s+['\"]?([\w][\w\s\-]*?)['\"]?(?:\s*$|\s*[,.])|"""
    r"""template\s+['\"]?([\w][\w\s\-]*?)['\"]?(?:\s*$|\s+(?:on|for|to|and))|"""
    r"""['\"]([^'"]+)['\"]""",
    re.IGNORECASE,
)


def _extract_template_name(message: str) -> str | None:
    """Extract the template name from a save/replay message.

    Handles:
      "save this as a template called 'Monthly Sales Review'"
      "replay my 'Q4 Analysis' template"
      "create a template named customer segments"
    """
    m = _TEMPLATE_NAME_RE.search(message)
    if not m:
        return None
    name = next((g for g in m.groups() if g), None)
    return name.strip() if name else None


# ---------------------------------------------------------------------------
# Prediction preset patterns — named quick-fill scenarios for VP dashboard
# ---------------------------------------------------------------------------

_PRESET_SAVE_PATTERNS = re.compile(
    r"\b("
    r"save\s+(?:this\s+as\s+a?\s*)?(?:prediction\s+)?preset|"
    r"add\s+(?:a\s+)?(?:prediction\s+)?preset(?:\s+called|\s+named)?|"
    r"create\s+(?:a\s+)?(?:prediction\s+)?preset(?:\s+called|\s+named)?|"
    r"make\s+(?:a\s+)?preset(?:\s+called|\s+named)?|"
    r"save\s+(?:this\s+as\s+)?(?:a\s+)?(?:named\s+)?scenario(?:\s+called|\s+named)?|"
    r"add\s+(?:a\s+)?(?:named\s+)?scenario(?:\s+called|\s+named)?|"
    r"bookmark\s+(?:this\s+)?(?:as\s+(?:a\s+)?)?preset|"
    r"quick\s+scenario\s+(?:called|named)"
    r")\b",
    re.IGNORECASE,
)

_PRESET_LIST_PATTERNS = re.compile(
    r"\b("
    r"(?:show|list|what|view)\s+(?:my\s+)?(?:prediction\s+)?presets?|"
    r"(?:show|list)\s+(?:saved\s+)?scenarios?|"
    r"what\s+presets?\s+(?:do\s+I\s+have|are\s+saved)|"
    r"(?:saved|existing)\s+presets?"
    r")\b",
    re.IGNORECASE,
)

_PRESET_NAME_RE = re.compile(
    r"(?:called|named)\s+['\"]?([A-Za-z0-9][A-Za-z0-9 _\-]*?)['\"]?\s*"
    r"(?=\s+with\s+|\s*:\s*[A-Za-z]|\s*,\s*[A-Za-z]|$)",
    re.IGNORECASE,
)
_PRESET_KV_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s,;]+)",
)

_SDK_PATTERNS = re.compile(
    r"\b("
    r"(?:generate|create|download|get|give\s+me|build)\s+(?:a\s+|the\s+)?(?:python|javascript|js)?\s*sdk|"
    r"(?:python|javascript|js)\s+sdk\s+(?:for|of)\s+(?:my|this)?(?:\s+model)?|"
    r"(?:generate|download|get)\s+(?:the\s+)?(?:python|javascript|js)\s+(?:client|library|module|sdk)|"
    r"sdk\s+(?:for|to)\s+(?:call|use|integrate|consume)\s+(?:my|this|the)\s+(?:model|api|prediction)|"
    r"client\s+library\s+for\s+(?:my|this|the)\s+(?:model|api)|"
    r"how\s+(?:do\s+(?:I|developers?)|can\s+(?:my\s+)?developers?)\s+(?:use|integrate|consume|call)\s+(?:my|this|the)\s+(?:model|api|prediction\s+api)|"
    r"(?:make|create)\s+(?:it|this)\s+easy\s+for\s+(?:my\s+)?developers?|"
    r"developer\s+sdk|developer\s+client\s+library"
    r")\b",
    re.IGNORECASE,
)

_PORTFOLIO_PATTERNS = re.compile(
    r"\b("
    r"(?:show|list|give\s+me|display)\s+(?:all\s+(?:my\s+)?|my\s+)?(?:models|projects|deployments|predictions)|"
    r"portfolio(?:\s+overview|\s+summary|\s+view)?|"
    r"(?:all|my)\s+(?:my\s+)?(?:prediction\s+)?(?:projects|models)\s+(?:overview|summary|status)|"
    r"(?:compare|overview)\s+(?:all\s+)?(?:my\s+)?(?:projects|models|deployments)|"
    r"which\s+(?:project|model)\s+is\s+(?:doing\s+)?best|"
    r"cross[\s-]project\s+(?:view|overview|summary|comparison)|"
    r"how\s+(?:many|are)\s+(?:my\s+)?(?:all\s+)?(?:projects|models|deployments)|"
    r"all\s+my\s+(?:prediction\s+)?work"
    r")\b",
    re.IGNORECASE,
)

_RATE_LIMIT_PATTERNS = re.compile(
    r"\b("
    r"(?:set|add|enable|configure|apply|create)\s+(?:a\s+)?rate\s+(?:limit|limiting)|"
    r"rate\s+limit(?:ing)?(?:\s+(?:my|the|this))?\s+(?:model|api|endpoint|deployment)?|"
    r"limit(?:\s+(?:to|the))?\s+(?:\d+\s+)?requests?\s+(?:per|a)\s+minute|"
    r"(?:requests?\s+per\s+minute|rpm)\s+limit|"
    r"(?:set|add|apply|configure|enable|create)\s+(?:a\s+)?monthly\s+quota|"
    r"monthly\s+(?:prediction\s+)?quota|"
    r"limit\s+(?:to\s+)?\d+\s+predictions?\s+(?:per|a)\s+month|"
    r"prediction\s+(?:usage\s+)?(?:quota|limit|cap)|"
    r"usage\s+(?:quota|limit|cap)|"
    r"quota\s+status|check\s+(?:my\s+)?quota|how\s+many\s+predictions?\s+(?:left|remaining)|"
    r"disable\s+rate\s+limit|remove\s+rate\s+limit|turn\s+off\s+rate\s+limit|"
    r"disable\s+(?:monthly\s+)?quota|remove\s+(?:monthly\s+)?quota"
    r")\b",
    re.IGNORECASE,
)

_RATE_LIMIT_NUMBER_RE = re.compile(
    r"\b(\d+)\s*(?:requests?\s+per\s+minute|rpm)\b", re.IGNORECASE
)
_QUOTA_NUMBER_RE = re.compile(
    r"\b(\d+)\s+predictions?\s+(?:per|a)\s+month\b", re.IGNORECASE
)
_DISABLE_RATE_RE = re.compile(
    r"\b(disable|remove|turn\s+off|clear)\s+(?:the\s+)?rate\s+limit", re.IGNORECASE
)
_DISABLE_QUOTA_RE = re.compile(
    r"\b(disable|remove|turn\s+off|clear)\s+(?:the\s+)?(?:monthly\s+)?quota",
    re.IGNORECASE,
)

_SLA_PATTERNS = re.compile(
    r"\b("
    r"(?:show|check|view|get|what(?:'s|\s+is)?)\s+(?:\w+\s+){0,3}(?:prediction\s+)?latency|"
    r"(?:prediction\s+)?latency\s+(?:stats|statistics|metrics|numbers|data|report)|"
    r"how\s+fast\s+is\s+(?:my\s+)?(?:model|api|endpoint|deployment)|"
    r"(?:model|api|endpoint)\s+speed|"
    r"p(?:95|99|50)\s+latency|latency\s+p(?:95|99|50)|"
    r"(?:is\s+(?:my\s+)?(?:model|api|endpoint)\s+(?:within|meeting|hitting))\s+sla|"
    r"sla\s+(?:status|check|report|metrics|monitoring)|"
    r"response\s+time(?:\s+(?:stats|statistics|metrics|data))?|"
    r"how\s+long\s+(?:does|do)\s+(?:it|predictions?)\s+take|"
    r"prediction\s+speed\s+(?:stats|report|metrics)"
    r")\b",
    re.IGNORECASE,
)

# Keywords that trigger quota alert configuration via chat
_QUOTA_ALERT_PATTERNS = re.compile(
    r"\b("
    r"alert\s+(?:me\s+)?when\s+(?:i\s+)?(?:hit|reach|use)\s+(?:[\w%]+\s+){0,4}quota|"
    r"notify\s+(?:me\s+)?when\s+(?:my\s+)?quota\s+is\s+(?:almost|nearly|close\s+to)\s+(?:full|exhausted|used)|"
    r"quota\s+(?:usage\s+)?alert|quota\s+(?:usage\s+)?warning|"
    r"set\s+(?:a\s+)?quota\s+(?:alert|warning|notification)\s+(?:at|threshold|to)|"
    r"configure\s+quota\s+(?:alert|warning|notification)|"
    r"(?:alert|warn|notify)\s+(?:me\s+)?(?:at|when\s+quota\s+(?:reaches|hits))\s+\d+|"
    r"(?:disable|remove|turn\s+off|clear)\s+(?:the\s+)?quota\s+alert|"
    r"quota\s+(?:alert\s+)?threshold"
    r")\b",
    re.IGNORECASE,
)

_QUOTA_ALERT_PCT_RE = re.compile(r"\b(\d+)\s*(?:%|percent)\b", re.IGNORECASE)
_DISABLE_QUOTA_ALERT_RE = re.compile(
    r"\b(disable|remove|turn\s+off|clear)\s+(?:the\s+)?quota\s+alert",
    re.IGNORECASE,
)

# Keywords that trigger class imbalance detection via chat
_CLASS_IMBALANCE_PATTERNS = re.compile(
    r"(?i)\b("
    r"class\s+imbalance|imbalanced?\s+(?:class(?:es)?|data(?:set)?)|"
    r"(?:data(?:set)?)\s+is\s+imbalanced?|"
    r"imbalanced?\s+target|skewed\s+(?:class(?:es)?|target|data)|"
    r"rare\s+(?:class|event|case|positive|category)|"
    r"minority\s+class|majority\s+class|"
    r"my\s+(?:positive|negative|target)\s+class\s+is\s+(?:rare|small|low|tiny|few)|"
    r"(?:only|just)\s+\d+\s*%?\s+(?:are\s+)?(?:positive|negative|true|churn|fraud)|"
    r"handle\s+imbalance|fix\s+imbalance|deal\s+with\s+imbalance|"
    r"class[\s_]weight|smote|oversample|undersample|balance\s+(?:my\s+)?(?:class(?:es)?|data|target)|"
    r"unbalanced\s+(?:class(?:es)?|data|target)|"
    r"is\s+(?:my\s+)?(?:data|target|dataset)\s+(?:balanced|imbalanced|unbalanced)|"
    r"check\s+(?:for\s+)?(?:class\s+)?imbalance"
    r")\b",
    re.IGNORECASE,
)


def _extract_preset_definition(message: str) -> dict | None:
    """Extract preset name and feature key=value pairs from a natural language message.

    Returns {name, feature_values} or None if not parseable.
    Examples:
      "save as a preset called Best Case: Region=East, Units=500"
      -> {name: "Best Case", feature_values: {"Region": "East", "Units": 500}}
    """
    name_m = _PRESET_NAME_RE.search(message)
    if not name_m:
        return None
    name = name_m.group(1).strip().rstrip(":-,")
    if not name:
        return None

    kv_pairs = _PRESET_KV_RE.findall(message)
    if not kv_pairs:
        return None

    feature_values: dict = {}
    for key, val in kv_pairs:
        try:
            feature_values[key] = float(val) if "." in val else int(val)
        except ValueError:
            feature_values[key] = val

    return {"name": name, "feature_values": feature_values}


# ---------------------------------------------------------------------------
# Dataset ranking patterns — "which customers are most likely to churn?"
# ---------------------------------------------------------------------------

_RANKED_PRED_PATTERNS = re.compile(
    r"(?i)(?:"
    r"which\s+\w+(?:\s+\w+)?\s+(?:are|is|have|has)\s+(?:the\s+)?(?:most|highest|lowest|least)\s+(?:likely|predicted|probable|expected)\b|"
    r"(?:rank|sort|prioritize|order)\s+(?:by|the|my|all)?\s*(?:predicted|prediction|model|probability)\b|"
    r"(?:show|find|get)\s+(?:me\s+)?(?:the\s+)?(?:top|bottom)\s+\d+\b|"
    r"(?:top|bottom)\s+\d+\s+(?:by\s+)?(?:prediction|predicted|probability|confidence|score)\b|"
    r"(?:most|least)\s+(?:at\s+risk|likely\s+to|probable|confident)\b|"
    r"(?:who|which)\s+(?:is|are)\s+(?:most|least|at\s+high|at\s+low)\s+(?:risk|likely)\b|"
    r"(?:best|worst|highest|lowest)\s+(?:\d+\s+)?(?:opportunities?|predictions?|candidates?|accounts?|customers?|records?)\b|"
    r"apply\s+(?:the\s+)?model\s+to\s+(?:all|the|my|entire)(?:\s+\w+)?\s*(?:data|dataset|rows)\b"
    r")",
    re.IGNORECASE,
)


_COHORT_PATTERNS = re.compile(
    r"(?i)(?:"
    r"who\s+(?:are|is)\s+(?:the|these|those|my)?\s*(?:top|highest|at.risk|ranked|predicted)\b|"
    r"(?:what\s+do|what\s+does)\s+(?:the|these|those|my)?\s*(?:top|ranked|highest|at.risk)\s+(?:\d+\s+)?(?:predictions?|records?|customers?|accounts?|rows?)\s+(?:have\s+in\s+common|look\s+like|share)\b|"
    r"(?:profile|characterize|describe|segment|group)\s+(?:the|my|these)?\s*(?:top|ranked|highest|at.risk|predicted)(?:\s+(?:\d+|\w+))?\s*(?:predictions?|records?|customers?|accounts?|rows?)\b|"
    r"(?:common\s+(?:traits?|characteristics?|features?|patterns?)\s+(?:of|among|in)\s+(?:the\s+)?(?:top|ranked|highest|at.risk))\b|"
    r"(?:what\s+(?:do|does)\s+(?:the|my)\s+)?(?:top|at.risk|highest.scoring)\s+(?:\d+\s+)?(?:predictions?|records?|customers?|accounts?)\s+(?:have\s+in\s+common|look\s+like)\b|"
    r"(?:cohort|segment)\s+(?:analysis|profile)\s+(?:of|on|for)?\s*(?:the|my)?\s*(?:ranked|top|predicted)\b|"
    r"(?:tell\s+me\s+about|describe|explain)\s+(?:the|these|those|my)?\s*(?:top|ranked|at.risk)\s+(?:\d+\s+)?(?:predictions?|records?|customers?|accounts?|rows?)\b|"
    r"(?:are\s+there\s+(?:any\s+)?(?:common|shared|similar)\s+(?:traits?|characteristics?|patterns?)\s+(?:among|in|for))\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Batch schedule patterns — "schedule daily predictions at 9am"
# ---------------------------------------------------------------------------

_SCHEDULE_PATTERNS = re.compile(
    r"(?i)(?:"
    r"schedule\s+(?:daily|weekly|monthly|batch|automatic|recurring|a)?\s*(?:batch\s+)?predictions?\b|"
    r"(?:set\s+up|create|configure|add)\s+(?:a\s+)?(?:daily|weekly|monthly|batch|automatic|recurring)?\s*(?:batch\s+)?(?:prediction\s+)?schedule\b|"
    r"(?:run|predict|score)\s+(?:my\s+)?(?:model|data|batch|predictions?)\s+(?:every|each)\s+(?:day|week|month|\w+day)\b|"
    r"batch\s+predictions?\s+every\s+(?:day|week|month)\b|"
    r"automatic(?:ally)?\s+(?:run|predict|score)\s+(?:every|each|daily|weekly|monthly)\b|"
    r"(?:run|predict)\s+(?:my\s+)?(?:model|batch)\s+at\s+\d+\b|"
    r"(?:show|list|view)\s+(?:my\s+)?(?:batch\s+)?schedules?\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# A/B test patterns — "how is my A/B test going?", "promote the challenger"
# Covers: status check, promote challenger, end test
# Does NOT overlap with _READINESS_PATTERNS or _ALERTS_PATTERNS
# ---------------------------------------------------------------------------

_AB_TEST_PATTERNS = re.compile(
    r"(?i)(?:"
    r"(?:how\s+is|check|show|view|get|what\s+(?:are|is))\s+(?:my\s+)?(?:a\s*/\s*b|ab|split|champion.challenger)\s+test\b|"
    r"a\s*/\s*b\s+test\s+(?:results?|status|progress|update)\b|"
    r"(?:is|are)\s+(?:the\s+)?challenger\s+(?:doing|performing|beating|better)\b|"
    r"(?:compare|versus|vs)\s+(?:the\s+)?(?:champion|challenger)\s+model\b|"
    r"promote\s+(?:the\s+)?challenger\b|"
    r"make\s+(?:the\s+)?challenger\s+(?:the\s+)?(?:production|live|champion|main)\s+model\b|"
    r"(?:end|stop|finish|close)\s+(?:the\s+)?(?:a\s*/\s*b|ab|split)\s+test\b|"
    r"(?:traffic\s+split|split\s+traffic|prediction\s+split)\s+(?:results?|status)?\b"
    r")",
    re.IGNORECASE,
)

_AB_PROMOTE_RE = re.compile(
    r"(?i)\b(promote|make\s+(?:the\s+)?challenger\s+(?:the\s+)?(?:production|live|champion|main))\b"
)
_AB_END_RE = re.compile(
    r"(?i)\b(end|stop|finish|close)\s+(?:the\s+)?(?:a\s*/\s*b|ab|split)\s+test\b"
)

# Webhook event history — "what webhooks fired recently?", "show webhook log", etc.
_WEBHOOK_HISTORY_PATTERNS = re.compile(
    r"(?i)(?:"
    r"(?:what|which|show|list|view|get)\s+(?:my\s+)?(?:webhooks?|webhook\s+events?|webhook\s+notifications?|webhook\s+(?:fire|fired|trigger|triggered))\b|"
    r"webhook\s+(?:histor(?:y|ies)?|log|events?|record|activit(?:y|ies)?|fire|trigger|notification)\b|"
    r"(?:recent|latest|last)\s+webhook\b|"
    r"(?:did|have)\s+(?:any\s+)?webhooks?\s+(?:fire|fired|trigger|triggered|sent|gone\s+off)\b|"
    r"webhook\s+(?:status|report|summary)\b|"
    r"(?:show|check|view)\s+(?:my\s+)?webhook\s+(?:histor(?:y|ies)?|log|activit(?:y|ies)?|events?)\b"
    r")",
    re.IGNORECASE,
)

_DOW_NAMES: dict[str, int] = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}


def _extract_schedule_params(message: str) -> dict:
    """Parse frequency, hour, minute, day_of_week/month from a natural-language schedule request."""
    msg = message.lower()

    # Frequency + day-of-week detection
    frequency = "daily"
    day_of_week: int | None = None
    day_of_month: int | None = None

    if re.search(r"\b(weekly|every\s+week|each\s+week)\b", msg):
        frequency = "weekly"
    elif re.search(r"\b(monthly|every\s+month|each\s+month|once\s+a\s+month)\b", msg):
        frequency = "monthly"
    else:
        # Check for explicit weekday names — implies weekly
        for name, idx in _DOW_NAMES.items():
            if re.search(r"\b" + name + r"\b", msg):
                frequency = "weekly"
                day_of_week = idx
                break

    if frequency == "weekly" and day_of_week is None:
        day_of_week = 0  # default Monday

    if frequency == "monthly":
        dom_m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", msg)
        if dom_m:
            day_of_month = max(1, min(28, int(dom_m.group(1))))
        if day_of_month is None:
            day_of_month = 1  # default 1st

    # Time extraction: "at 9am", "at 9:30", "9pm", "14:00"
    run_hour = 9
    run_minute = 0
    time_m = re.search(
        r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", msg, re.IGNORECASE
    )
    if not time_m:
        time_m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", msg, re.IGNORECASE)
    if time_m:
        h = int(time_m.group(1))
        m = int(time_m.group(2)) if time_m.group(2) else 0
        ampm = (time_m.group(3) or "").lower()
        if ampm == "pm" and h != 12:
            h += 12
        elif ampm == "am" and h == 12:
            h = 0
        run_hour = max(0, min(23, h))
        run_minute = max(0, min(59, m))

    return {
        "frequency": frequency,
        "run_hour": run_hour,
        "run_minute": run_minute,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
    }


def _build_schedule_description(
    frequency: str,
    run_hour: int,
    run_minute: int,
    day_of_week: int | None,
    day_of_month: int | None,
) -> str:
    """Return a plain-English description of a schedule."""
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    hh = str(run_hour).zfill(2)
    mm = str(run_minute).zfill(2)
    if frequency == "daily":
        return f"Every day at {hh}:{mm} UTC"
    if frequency == "weekly":
        dow_name = day_names[day_of_week] if day_of_week is not None else "Monday"
        return f"Every {dow_name} at {hh}:{mm} UTC"
    # monthly
    dom = day_of_month or 1
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(dom if dom <= 20 else dom % 10, "th")
    return f"Monthly on the {dom}{suffix} at {hh}:{mm} UTC"


def _detect_ranked_pred_request(message: str) -> dict:
    """Extract n (number of rows) and direction (highest/lowest) from a ranking request.

    Returns {"n": int, "direction": str}.
    """
    # Extract N from "top 20", "bottom 10", "top N"
    n_match = re.search(r"\b(top|bottom)\s+(\d+)\b", message, re.IGNORECASE)
    n = int(n_match.group(2)) if n_match else 20
    n = max(1, min(n, 100))  # Cap at 100 rows

    direction_word = n_match.group(1).lower() if n_match else None

    # Determine direction
    lowest_hints = re.compile(
        r"\b(lowest|least|worst|bottom|minimum|min|at\s+lowest)\b", re.IGNORECASE
    )
    highest_hints = re.compile(
        r"\b(highest|most|best|top|maximum|max|at\s+highest)\b", re.IGNORECASE
    )

    if direction_word == "bottom" or (
        lowest_hints.search(message) and not highest_hints.search(message)
    ):
        direction = "lowest"
    else:
        direction = "highest"

    return {"n": n, "direction": direction}


def _detect_sensitivity_request(
    message: str, feature_names: list[str], feature_means: dict
) -> dict | None:
    """Extract feature name and sweep range from a sensitivity message.

    Returns {"feature": str, "min_val": float, "max_val": float, "n_steps": int}
    or None if no numeric feature can be resolved.
    """
    msg_lower = message.lower()

    # Longest-match scan for a mentioned feature
    feature: str | None = None
    for cand in sorted(feature_names, key=len, reverse=True):
        c_low = cand.lower()
        c_ns = cand.lower().replace("_", " ")
        if c_low in msg_lower or c_ns in msg_lower:
            feature = cand
            break
    if feature is None:
        # Fall back to the first numeric-looking feature
        for f in feature_names:
            if f in feature_means and isinstance(feature_means[f], (int, float)):
                feature = f
                break
    if feature is None:
        return None

    # Extract explicit range "from X to Y" or "between X and Y" or "X to Y"
    range_match = re.search(
        r"\b(?:from\s+)?(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)\b",
        message,
        re.IGNORECASE,
    )
    if range_match:
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
    else:
        # Default: ± 50% around the training mean for this feature
        mean_val = float(feature_means.get(feature, 1.0))
        half = abs(mean_val) * 0.5 or 1.0
        min_val = max(0.0, round(mean_val - half, 4))
        max_val = round(mean_val + half, 4)

    # Extract step count "in N steps" or "N steps"; default 10
    n_steps = 10
    step_match = re.search(r"\b(\d+)\s*steps?\b", message, re.IGNORECASE)
    if step_match:
        n = int(step_match.group(1))
        if 3 <= n <= 50:
            n_steps = n

    return {
        "feature": feature,
        "min_val": min_val,
        "max_val": max_val,
        "n_steps": n_steps,
    }


# Matches "Key = Value", "Key: Value", "Key is Value" patterns in a message
_KV_PAIR_RE = re.compile(
    r"\b([A-Za-z_][\w\s]{0,30}?)\s*(?:=|:|\s+is\s+|\s+equals?\s+|\s+of\s+)\s*"
    r"(['\"]?)([A-Za-z0-9_][\w\.\-]*)\2",
    re.IGNORECASE,
)


def _extract_multi_feature_prediction(
    message: str, feature_names: list[str]
) -> dict[str, object]:
    """Extract explicit feature=value pairs from a natural-language message.

    Returns a dict mapping feature name (as known in the model) to a typed value.
    Numeric strings are converted to float; everything else stays as str.
    Only features that are in *feature_names* are returned.
    """
    extracted: dict[str, object] = {}
    name_lower = {f.lower(): f for f in feature_names}
    name_nospace = {f.lower().replace("_", " "): f for f in feature_names}

    for m in _KV_PAIR_RE.finditer(message):
        raw_key = m.group(1).strip().lower()
        raw_val = m.group(3).strip()
        # Try exact match, then underscore→space variant
        canon = name_lower.get(raw_key) or name_nospace.get(raw_key)
        if canon is None:
            # Fuzzy: check if raw_key is a sub-word of any feature name
            for feat_lower, feat_orig in name_lower.items():
                if raw_key in feat_lower or feat_lower in raw_key:
                    canon = feat_orig
                    break
        if canon and canon not in extracted:
            try:
                extracted[canon] = float(raw_val)
            except ValueError:
                extracted[canon] = raw_val
    return extracted


def _detect_group_trend_request(message: str, df: "pd.DataFrame") -> dict | None:
    """Extract date_col, group_col, value_col from a group-trend message.

    Returns {"date_col": ..., "group_col": ..., "value_col": ...} or None.
    """
    from core.analyzer import detect_time_columns as _dtc

    msg_lower = message.lower()
    time_cols = _dtc(df)
    if not time_cols:
        return None
    date_col = time_cols[0]

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Exclude the date column from candidates
    cat_cols = [c for c in cat_cols if c != date_col]

    if not cat_cols or not numeric_cols:
        return None

    # Try to find a mentioned categorical column (longest-match first)
    group_col: str | None = None
    for c in sorted(cat_cols, key=len, reverse=True):
        if c.lower() in msg_lower or c.replace("_", " ").lower() in msg_lower:
            group_col = c
            break
    if group_col is None:
        group_col = cat_cols[0]

    # Try to find a mentioned numeric column (longest-match first)
    value_col: str | None = None
    for c in sorted(numeric_cols, key=len, reverse=True):
        if c.lower() in msg_lower or c.replace("_", " ").lower() in msg_lower:
            value_col = c
            break
    if value_col is None:
        value_col = numeric_cols[0]

    return {"date_col": date_col, "group_col": group_col, "value_col": value_col}


def _detect_histogram_col(message: str, df: "pd.DataFrame") -> str | None:
    """Extract the numeric column to histogram from the user message.

    Returns the column name or None if no numeric column found.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None
    msg_lower = message.lower()
    for col in sorted(numeric_cols, key=len, reverse=True):
        if col.lower() in msg_lower or col.replace("_", " ").lower() in msg_lower:
            return col
    return numeric_cols[0]


def _detect_topn_request(message: str, df: "pd.DataFrame") -> dict | None:
    """Extract sort column, n, and direction from the user message.

    Returns dict with sort_col, n, ascending  — or None if cannot be resolved.
    """
    # Detect direction: ascending = bottom/lowest/worst/smallest/fewest/least
    ascending = bool(
        re.search(
            r"\b(bottom|lowest|worst|smallest|fewest|least|minimum)\b",
            message,
            re.IGNORECASE,
        )
    )

    # Extract n (numeric or word)
    n = 10  # default
    m_digit = re.search(r"\b(\d+)\b", message)
    if m_digit:
        candidate = int(m_digit.group(1))
        if 1 <= candidate <= 50:
            n = candidate
    else:
        for word, val in _TOPN_N_WORDS.items():
            if re.search(r"\b" + word + r"\b", message, re.IGNORECASE):
                n = val
                break

    # Identify numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None

    # Try to match a column name from the message
    msg_lower = message.lower()
    sort_col = None
    for col in numeric_cols:
        if col.lower() in msg_lower or col.replace("_", " ").lower() in msg_lower:
            sort_col = col
            break

    # Fallback: pick first numeric column
    if sort_col is None:
        sort_col = numeric_cols[0]

    return {"sort_col": sort_col, "n": n, "ascending": ascending}


def _detect_whatif_request(
    message: str, feature_names: list[str]
) -> dict[str, object] | None:
    """Extract a feature name and new value from a natural-language what-if message.

    Uses a feature-name-first approach: for each known feature, searches the
    message for an associated value using targeted patterns. This avoids greedy
    regex captures that would include context phrases like "what if".

    Handles patterns like:
      "what if revenue was 500?"
      "what happens if I double the price?"
      "if quantity were 10, what would be predicted?"
      "change margin to 0.3"
      "predict with discount = 20"

    Returns {"feature": str, "new_value": float|str, "original_phrase": str} or None.
    """
    msg_lower = message.lower()
    # Build lookup: both underscore and space versions map to original feature name
    feat_variants: list[tuple[str, str]] = []
    for feat in feature_names:
        feat_variants.append((feat.lower(), feat))  # e.g. "total_revenue" → original
        spaced = feat.lower().replace("_", " ")
        if spaced != feat.lower():
            feat_variants.append((spaced, feat))  # e.g. "total revenue" → original

    _val_pattern = r"([\"']?[\w.,%-]+[\"']?)"

    for feat_key, feat_orig in feat_variants:
        if feat_key not in msg_lower:
            continue

        feat_escaped = re.escape(feat_key)

        # Pattern A: "<feat> was/is/were/becomes/equals/to <value>"
        ma = re.search(
            feat_escaped
            + r"\s+(?:was|is|were|becomes?|equals?|changed?\s+to|set\s+to)\s+"
            + _val_pattern,
            message,  # original case for value extraction
            re.IGNORECASE,
        )
        if ma:
            return _build_whatif_result(feat_orig, ma.group(1))

        # Pattern B: "change <feat> to <value>"
        mb = re.search(
            r"\bchange\s+" + feat_escaped + r"\s+to\s+" + _val_pattern,
            message,
            re.IGNORECASE,
        )
        if mb:
            return _build_whatif_result(feat_orig, mb.group(1))

        # Pattern C: "<feat> = <value>" or "<feat>: <value>"
        mc = re.search(
            feat_escaped + r"\s*[=:]\s*" + _val_pattern,
            message,
            re.IGNORECASE,
        )
        if mc:
            return _build_whatif_result(feat_orig, mc.group(1))

    # --- Fallback: detect "double/halve the <feature>" ---
    multiplier_map = {
        "doubl": 2.0,  # matches "double", "doubled"
        "tripl": 3.0,  # matches "triple", "tripled"
        "halv": 0.5,  # matches "halve", "halved"
        "half": 0.5,
    }
    for prefix, mult in multiplier_map.items():
        if prefix not in msg_lower:
            continue
        pm = re.search(
            r"\b" + prefix + r"\w*\s+(?:the\s+)?(\w[\w\s]*?)(?:\s*[?!,.]|$)",
            msg_lower,
            re.IGNORECASE,
        )
        if not pm:
            continue
        candidate = pm.group(1).strip()
        # Match candidate to feature names / space variants
        for feat_key, feat_orig in feat_variants:
            if feat_key == candidate or feat_key in candidate or candidate in feat_key:
                return {
                    "feature": feat_orig,
                    "new_value": f"__multiply__{mult}",
                    "original_phrase": f"{prefix} {feat_orig}",
                }

    return None


def _build_whatif_result(feat: str, raw_val: str) -> dict[str, object]:
    """Convert a raw string value into a typed result dict."""
    val_clean = raw_val.strip("\"'").replace(",", "")
    try:
        parsed: float | str = float(val_clean)
    except ValueError:
        parsed = raw_val.strip("\"'")
    return {
        "feature": feat,
        "new_value": parsed,
        "original_phrase": f"{feat} → {parsed}",
    }


class ChatMessage(BaseModel):
    message: str


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _compute_readiness(
    run: ModelRun, dataset: Dataset | None, feature_set: FeatureSet | None
) -> dict:
    """Inline model readiness calculation (same logic as /api/models/{id}/readiness)."""
    metrics = json.loads(run.metrics) if run.metrics else {}
    problem_type = (feature_set.problem_type if feature_set else None) or "regression"
    row_count = dataset.row_count if dataset else 0
    feature_count = (
        len(json.loads(feature_set.column_mapping or "{}")) if feature_set else 0
    )

    checks: list[dict] = []
    total_points = 0
    earned_points = 0

    # Training complete
    total_points += 10
    earned_points += 10
    checks.append(
        {
            "id": "training_complete",
            "label": "Training completed",
            "passed": True,
            "weight": 10,
        }
    )

    # Sufficient data
    total_points += 20
    data_ok = row_count >= 100
    earned_points += 20 if data_ok else (10 if row_count >= 50 else 0)
    checks.append(
        {
            "id": "sufficient_data",
            "label": f"Sufficient data ({row_count} rows)",
            "passed": data_ok,
            "weight": 20,
        }
    )

    # Accuracy
    total_points += 30
    if problem_type == "regression":
        r2 = metrics.get("r2", 0.0)
        perf_ok = r2 >= 0.7
        earned_points += 30 if perf_ok else (15 if r2 >= 0.5 else 0)
        checks.append(
            {
                "id": "accuracy",
                "label": f"R² = {r2:.3f} (threshold: 0.70)",
                "passed": perf_ok,
                "weight": 30,
            }
        )
    else:
        acc = metrics.get("accuracy", 0.0)
        perf_ok = acc >= 0.8
        earned_points += 30 if perf_ok else (15 if acc >= 0.65 else 0)
        checks.append(
            {
                "id": "accuracy",
                "label": f"Accuracy = {acc:.1%} (threshold: 80%)",
                "passed": perf_ok,
                "weight": 30,
            }
        )

    # Features
    total_points += 15
    has_features = feature_count > 1
    earned_points += 15 if has_features else 5
    checks.append(
        {
            "id": "features",
            "label": f"{feature_count} features used",
            "passed": has_features,
            "weight": 15,
        }
    )

    # Data quality
    total_points += 15
    profile = json.loads(dataset.profile or "{}") if dataset else {}
    missing_pct = profile.get("missing_percentage", 0.0)
    dq_ok = missing_pct < 10.0
    earned_points += 15 if dq_ok else (8 if missing_pct < 30.0 else 0)
    checks.append(
        {
            "id": "data_quality",
            "label": f"Data quality ({missing_pct:.1f}% missing)",
            "passed": dq_ok,
            "weight": 15,
        }
    )

    # Selected
    total_points += 10
    earned_points += 10 if run.is_selected else 0
    checks.append(
        {
            "id": "selected",
            "label": "Marked as preferred model",
            "passed": run.is_selected,
            "weight": 10,
        }
    )

    score = round((earned_points / total_points) * 100) if total_points > 0 else 0
    verdict = (
        "ready" if score >= 85 else ("needs_attention" if score >= 60 else "not_ready")
    )

    return {
        "model_run_id": run.id,
        "algorithm": run.algorithm,
        "score": score,
        "verdict": verdict,
        "checks": checks,
        "problem_type": problem_type,
    }


def _compute_drift(deployment: Deployment, logs: list) -> dict:
    """Compute prediction drift inline (same logic as GET /api/deploy/{id}/drift)."""
    WINDOW = 10
    logs_sorted = sorted(logs, key=lambda log: log.created_at)

    if len(logs_sorted) < WINDOW * 2:
        return {
            "deployment_id": deployment.id,
            "status": "insufficient_data",
            "drift_score": None,
            "explanation": (
                f"Need at least {WINDOW * 2} predictions to detect drift "
                f"(currently {len(logs_sorted)})."
            ),
            "problem_type": deployment.problem_type,
        }

    baseline = logs_sorted[:WINDOW]
    recent = logs_sorted[-WINDOW:]
    problem_type = deployment.problem_type or "regression"

    if problem_type == "regression":
        b_vals = [
            log.prediction_numeric
            for log in baseline
            if log.prediction_numeric is not None
        ]
        r_vals = [
            log.prediction_numeric
            for log in recent
            if log.prediction_numeric is not None
        ]
        if not b_vals or not r_vals:
            return {
                "deployment_id": deployment.id,
                "status": "insufficient_data",
                "drift_score": None,
                "explanation": "No numeric values.",
                "problem_type": problem_type,
            }
        b_mean = sum(b_vals) / len(b_vals)
        r_mean = sum(r_vals) / len(r_vals)
        b_std = (sum((v - b_mean) ** 2 for v in b_vals) / len(b_vals)) ** 0.5
        z = abs(r_mean - b_mean) / (b_std + 1e-9)
        drift_score = min(100, int(z * 33))
        status = (
            "stable" if z < 1.0 else ("mild_drift" if z < 2.0 else "significant_drift")
        )
        explanation = (
            f"Prediction mean shifted from {b_mean:.3f} to {r_mean:.3f} "
            f"(z={z:.1f}). Status: {status.replace('_', ' ')}."
        )
    else:

        def _dist(ls: list) -> dict[str, float]:
            counts: dict[str, int] = {}
            for log in ls:
                try:
                    label = str(json.loads(log.prediction))
                except (json.JSONDecodeError, TypeError):
                    label = "unknown"
                counts[label] = counts.get(label, 0) + 1
            total = sum(counts.values()) or 1
            return {k: v / total for k, v in counts.items()}

        b_dist = _dist(baseline)
        r_dist = _dist(recent)
        all_classes = set(b_dist) | set(r_dist)
        tvd = sum(abs(r_dist.get(c, 0) - b_dist.get(c, 0)) for c in all_classes) / 2
        drift_score = min(100, int(tvd * 200))
        status = (
            "stable"
            if tvd < 0.1
            else ("mild_drift" if tvd < 0.25 else "significant_drift")
        )
        explanation = (
            f"Class distribution TVD={tvd:.2f}. Status: {status.replace('_', ' ')}."
        )

    return {
        "deployment_id": deployment.id,
        "status": status,
        "drift_score": drift_score,
        "explanation": explanation,
        "problem_type": problem_type,
    }


def _compute_health(
    deployment: Deployment, run: ModelRun, feedback_records: list, all_logs: list
) -> dict:
    """Compute model health inline (same logic as GET /api/deploy/{id}/health).

    Returns health_score 0-100, status, and a human-readable summary suitable
    for injecting into the system prompt.
    """
    from datetime import UTC, datetime

    # Age
    age_days = 0
    age_score = 100
    if run and run.created_at:
        now = datetime.now(UTC).replace(tzinfo=None)
        age_days = max(0, (now - run.created_at).days)
        age_score = (
            100
            if age_days <= 30
            else (75 if age_days <= 60 else (50 if age_days <= 90 else 25))
        )

    # Feedback
    feedback_score = 100
    feedback_note = "no feedback yet"
    has_feedback = bool(feedback_records)
    if has_feedback:
        problem_type = deployment.problem_type or "regression"
        if problem_type == "regression":
            pairs = [
                (fb.actual_value, fb.prediction_log_id)
                for fb in feedback_records
                if fb.actual_value is not None and fb.prediction_log_id
            ]
            # Simplified: just use count-based heuristic if we can't get pairs
            if pairs:
                feedback_score = 75  # Moderate — we have data but can't compute inline without session
            else:
                feedback_score = 80
        else:
            rated = [fb for fb in feedback_records if fb.is_correct is not None]
            if rated:
                accuracy = sum(1 for fb in rated if fb.is_correct) / len(rated)
                feedback_score = (
                    100
                    if accuracy >= 0.9
                    else (75 if accuracy >= 0.75 else (50 if accuracy >= 0.6 else 20))
                )
                feedback_note = f"{accuracy:.1%} real-world accuracy from {len(rated)} feedback records"

    # Drift
    drift_health_score = 100
    has_drift_data = len(all_logs) >= 40
    if has_drift_data:
        logs_sorted = sorted(all_logs, key=lambda log: log.created_at)
        window = 20
        baseline_logs = logs_sorted[:window]
        recent_logs = logs_sorted[-window:]
        problem_type = deployment.problem_type or "regression"
        if problem_type == "regression":
            b_vals = [
                log.prediction_numeric
                for log in baseline_logs
                if log.prediction_numeric is not None
            ]
            r_vals = [
                log.prediction_numeric
                for log in recent_logs
                if log.prediction_numeric is not None
            ]
            if b_vals and r_vals:
                b_mean = sum(b_vals) / len(b_vals)
                r_mean = sum(r_vals) / len(r_vals)
                b_std = (sum((v - b_mean) ** 2 for v in b_vals) / len(b_vals)) ** 0.5
                z = abs(r_mean - b_mean) / (b_std + 1e-9)
                drift_health_score = 100 if z < 1.0 else (60 if z < 2.0 else 25)
        else:
            b_preds = [
                str(json.loads(log.prediction))
                for log in baseline_logs
                if log.prediction
            ]
            r_preds = [
                str(json.loads(log.prediction)) for log in recent_logs if log.prediction
            ]
            all_classes = set(b_preds + r_preds)
            if all_classes:
                b_n, r_n = len(b_preds) or 1, len(r_preds) or 1
                tvd = 0.5 * sum(
                    abs(b_preds.count(c) / b_n - r_preds.count(c) / r_n)
                    for c in all_classes
                )
                drift_health_score = 100 if tvd < 0.1 else (60 if tvd < 0.25 else 25)

    # Composite
    if has_feedback and has_drift_data:
        health_score = int(
            feedback_score * 0.4 + drift_health_score * 0.35 + age_score * 0.25
        )
    elif has_feedback:
        health_score = int(feedback_score * 0.55 + age_score * 0.45)
    elif has_drift_data:
        health_score = int(drift_health_score * 0.6 + age_score * 0.4)
    else:
        health_score = age_score

    status = (
        "healthy"
        if health_score >= 75
        else ("warning" if health_score >= 50 else "critical")
    )

    return {
        "deployment_id": deployment.id,
        "health_score": health_score,
        "status": status,
        "model_age_days": age_days,
        "algorithm": deployment.algorithm,
        "has_feedback_data": has_feedback,
        "has_drift_data": has_drift_data,
        "feedback_note": feedback_note,
    }


def _load_project_context(project_id: str, session: Session) -> dict:
    """Load the full project context needed for the state-aware system prompt."""
    dataset = session.exec(
        select(Dataset).where(Dataset.project_id == project_id)
    ).first()

    # Latest active feature set (most recently created)
    feature_set = None
    if dataset:
        feature_set = session.exec(
            select(FeatureSet)
            .where(
                FeatureSet.dataset_id == dataset.id,
                FeatureSet.is_active == True,  # noqa: E712
            )
            .order_by(FeatureSet.created_at.desc())  # type: ignore[arg-type]
        ).first()

    model_runs = list(
        session.exec(select(ModelRun).where(ModelRun.project_id == project_id)).all()
    )

    # Latest active deployment
    deployment = session.exec(
        select(Deployment)
        .where(
            Deployment.project_id == project_id,
            Deployment.is_active == True,  # noqa: E712
        )
        .order_by(Deployment.created_at.desc())  # type: ignore[arg-type]
    ).first()

    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "model_runs": model_runs,
        "deployment": deployment,
    }


@router.post("/{project_id}")
def send_message(
    project_id: str,
    body: ChatMessage,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get or create conversation
    statement = select(Conversation).where(Conversation.project_id == project_id)
    conversation = session.exec(statement).first()
    if not conversation:
        conversation = Conversation(project_id=project_id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

    messages = json.loads(conversation.messages)

    messages.append(
        {
            "role": "user",
            "content": body.message,
            "timestamp": _utcnow().isoformat(),
        }
    )

    # Load full project context for state-aware prompt
    ctx = _load_project_context(project_id, session)

    # Pass recent conversation messages for multi-turn context
    # Exclude the just-appended user message (last item) — Claude already gets
    # the full message list; we only want the preceding turns for the system prompt
    recent_for_context = messages[:-1][-6:]  # up to 3 exchanges before this message
    system_prompt = build_system_prompt(
        project,
        dataset=ctx["dataset"],
        feature_set=ctx["feature_set"],
        model_runs=ctx["model_runs"],
        deployment=ctx["deployment"],
        recent_messages=recent_for_context if recent_for_context else None,
    )

    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    client = anthropic.Anthropic()

    # Load active dataset filter (if any) — applied to all analysis DataFrame loads
    _active_filter_conditions: list | None = None
    if ctx["dataset"]:
        try:
            from models.dataset_filter import DatasetFilter as _DatasetFilter

            _af = session.exec(
                select(_DatasetFilter).where(
                    _DatasetFilter.dataset_id == ctx["dataset"].id
                )
            ).first()
            if _af:
                _active_filter_conditions = json.loads(_af.conditions)
                system_prompt += (
                    f"\n\n## Active Data Filter\n"
                    f"All analyses in this session are running on a filtered subset: "
                    f"**{_af.filter_summary}** "
                    f"({_af.filtered_rows:,} of {_af.original_rows:,} rows shown). "
                    "When referring to data statistics or results, note that they reflect "
                    "the filtered subset, not the full dataset."
                )
        except Exception:  # noqa: BLE001
            pass  # Filter load is nice-to-have; never crash chat

    # Capture dataset info for post-stream chart generation
    dataset = ctx["dataset"]
    dataset_file_path: str | None = dataset.file_path if dataset else None
    column_info: list = (
        json.loads(dataset.columns) if (dataset and dataset.columns) else []
    )

    # Check if this is a readiness-related question
    readiness_data: dict | None = None
    if _READINESS_PATTERNS.search(body.message):
        completed_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
        selected_run = next((mr for mr in completed_runs if mr.is_selected), None)
        target_run = selected_run or (completed_runs[-1] if completed_runs else None)
        if target_run:
            try:
                readiness_data = _compute_readiness(
                    target_run, ctx["dataset"], ctx["feature_set"]
                )
                # Inject readiness summary into system prompt so Claude can incorporate it
                score = readiness_data["score"]
                verdict = readiness_data["verdict"]
                passed = sum(1 for c in readiness_data["checks"] if c["passed"])
                total = len(readiness_data["checks"])
                system_prompt += (
                    f"\n\n## Model Readiness Check (just computed)\n"
                    f"Algorithm: {target_run.algorithm} | Score: {score}/100 | "
                    f"Verdict: {verdict.upper()} | Checks passed: {passed}/{total}\n"
                    "Reference this assessment in your response. Be specific about the score "
                    "and what the user should do next."
                )
            except Exception:  # noqa: BLE001
                pass  # Readiness check is nice-to-have; never crash chat

    # Check if this is a model improvement advice request
    improvement_event: dict | None = None
    if _IMPROVEMENT_PATTERNS.search(body.message) and ctx["model_runs"]:
        completed_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
        if completed_runs:
            try:
                from core.advisor import (
                    compute_improvement_suggestions as _compute_improve,
                )
                from core.trainer import detect_time_columns as _dtc

                selected_run = next(
                    (mr for mr in completed_runs if mr.is_selected), None
                )
                target_run = selected_run or completed_runs[-1]
                _metrics = json.loads(target_run.metrics or "{}")
                _algo = target_run.algorithm
                _problem_type = (
                    "classification"
                    if _algo in ctx.get("classification_algos", [])
                    else "regression"
                )
                # Detect problem_type from algorithm name
                if _algo.endswith("_classifier") or _algo in {
                    "logistic_regression",
                    "voting_classifier",
                    "stacking_classifier",
                }:
                    _problem_type = "classification"
                else:
                    _problem_type = "regression"

                _n_rows = ctx.get("dataset_row_count", 0) if ctx.get("dataset") else 0
                if hasattr(ctx.get("dataset"), "row_count"):
                    _n_rows = ctx["dataset"].row_count or 0

                _has_date = False
                if ctx.get("dataset") and ctx["dataset"].file_path:
                    try:
                        import pandas as _pd

                        _df_s = _pd.read_csv(ctx["dataset"].file_path, nrows=50)
                        _has_date = bool(_dtc(_df_s))
                    except Exception:  # noqa: BLE001
                        pass

                _n_features = 0
                if (
                    ctx.get("feature_set")
                    and ctx["feature_set"].target_column
                    and ctx.get("dataset")
                ):
                    try:
                        import json as _json

                        _cols = _json.loads(ctx["dataset"].columns or "[]")
                        _n_features = max(0, len(_cols) - 1)
                    except Exception:  # noqa: BLE001
                        pass

                improvement_event = _compute_improve(
                    metrics=_metrics,
                    algorithm=_algo,
                    problem_type=_problem_type,
                    n_features=_n_features,
                    n_rows=_n_rows,
                    has_date_col=_has_date,
                    date_col_used=bool(_metrics.get("date_col_used")),
                    n_weak_features=0,  # skip expensive model load in chat
                    is_ensemble=bool(_metrics.get("ensemble_type")),
                    is_calibrated=bool(_metrics.get("is_calibrated")),
                    imbalance_strategy=_metrics.get("imbalance_strategy"),
                    class_is_imbalanced=False,
                )
                improvement_event["run_id"] = target_run.id
                improvement_event["project_id"] = body.project_id
                _n = improvement_event["n_suggestions"]
                system_prompt += (
                    f"\n\n## Model Improvement Suggestions (just computed)\n"
                    f"Algorithm: {_algo} | {improvement_event['primary_metric_name']}: "
                    f"{round(improvement_event['primary_metric'], 2)}\n"
                    f"Found {_n} ranked improvement suggestion{'s' if _n != 1 else ''}:\n"
                    + "\n".join(
                        f"{s['rank']}. {s['title']}: {s['explanation']}"
                        for s in improvement_event["suggestions"]
                    )
                    + "\n\nPresent the top 2-3 suggestions to the user in a helpful, "
                    "encouraging tone. Each suggestion should explain what to do and why "
                    "it will help — reference the specific metric values above."
                )
            except Exception:  # noqa: BLE001
                pass  # Nice-to-have; never crash chat

    # Check if this is a model selection / criteria-comparison request
    model_select_event: dict | None = None
    if _MODEL_SELECT_PATTERNS.search(body.message) and ctx["model_runs"]:
        completed_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
        if len(completed_runs) >= 1:
            try:
                from core.advisor import compute_model_selection as _cms

                _criteria = _detect_selection_criteria(body.message)

                _runs_data = []
                for _mr in completed_runs:
                    _m = json.loads(_mr.metrics or "{}")
                    _pt = (
                        "classification"
                        if _mr.algorithm.endswith("_classifier")
                        or _mr.algorithm in {"logistic_regression"}
                        else "regression"
                    )
                    _runs_data.append(
                        {
                            "run_id": _mr.id,
                            "algorithm": _mr.algorithm,
                            "metrics": _m,
                            "problem_type": _pt,
                            "is_selected": _mr.is_selected,
                            "is_deployed": _mr.is_deployed,
                        }
                    )

                model_select_event = _cms(_runs_data, criteria=_criteria)
                model_select_event["project_id"] = body.project_id

                _winner = model_select_event.get("winner") or {}
                _crit_desc = model_select_event.get("criteria_description", _criteria)
                _n = model_select_event.get("n_runs", 0)
                _ranked = model_select_event.get("ranked_runs", [])
                system_prompt += (
                    f"\n\n## Model Selection Recommendation (criteria: {_criteria})\n"
                    f"Criteria: {_crit_desc}\n"
                    f"Winner: {_winner.get('algorithm_plain', '')} "
                    f"({_winner.get('primary_metric_name', '')}: "
                    f"{round((_winner.get('primary_metric', 0) or 0) * 100)}%)\n"
                    f"Compared {_n} completed model run{'s' if _n != 1 else ''}.\n"
                    + "\n".join(
                        f"{r['rank']}. {r['algorithm_plain']} "
                        f"(score: {round(r['score'] * 100)}%)"
                        for r in _ranked
                    )
                    + f"\n\nExplain to the user why {_winner.get('algorithm_plain', 'this model')} "
                    "is the best choice for their criteria. Use the criteria description and the "
                    "winner's 'why' field to narrate the recommendation. Keep it encouraging and "
                    "non-technical. If there's only one model, tell them it's the best available "
                    "and suggest training others for comparison."
                )
            except Exception:  # noqa: BLE001
                pass  # Nice-to-have; never crash chat

    # Goal-driven training: analyst specifies a target metric and AutoModeler tries algorithms
    goal_train_event: dict | None = None
    if _GOAL_TRAIN_PATTERNS.search(body.message) and ctx["feature_set"]:
        _fs = ctx["feature_set"]
        if _fs.target_column and ctx["dataset"] and ctx["dataset"].file_path:
            _goal_info = _extract_goal_target(
                body.message,
                _fs.problem_type or "regression",
            )
            if _goal_info:
                _goal_metric, _goal_target = _goal_info
                try:
                    from pathlib import Path as _Path

                    import pandas as _pd

                    from core.trainer import (
                        prepare_features as _prepare_features,
                    )
                    from core.trainer import (
                        run_goal_driven_training as _run_goal,
                    )

                    _df_goal = _pd.read_csv(ctx["dataset"].file_path)
                    _tfms = json.loads(_fs.transformations or "[]")
                    if _tfms:
                        from core.feature_engine import (
                            apply_transformations as _apply_tfms,
                        )

                        _df_goal, _ = _apply_tfms(_df_goal, _tfms)
                    _target_col = _fs.target_column
                    _feat_cols = [c for c in _df_goal.columns if c != _target_col]
                    _problem_type = _fs.problem_type or "regression"
                    _X, _y, _le = _prepare_features(
                        _df_goal, _feat_cols, _target_col, _problem_type
                    )
                    import uuid as _uuid

                    _gbase = f"goal_{_uuid.uuid4().hex[:8]}"
                    _mdir = _Path("data/deployments")
                    _mdir.mkdir(parents=True, exist_ok=True)
                    goal_train_event = _run_goal(
                        _X, _y, _problem_type, _goal_metric, _goal_target, _mdir, _gbase
                    )
                    goal_train_event["project_id"] = body.project_id
                    goal_train_event["target_col"] = _target_col

                    _metric_label = {
                        "r2": "R²",
                        "accuracy": "accuracy",
                        "f1": "F1 score",
                        "precision": "precision",
                        "recall": "recall",
                    }.get(_goal_metric, _goal_metric.upper())
                    _tgt_str = (
                        f"{_goal_target:.2f}"
                        if _goal_metric == "r2"
                        else f"{_goal_target * 100:.0f}%"
                    )
                    system_prompt += (
                        f"\n\n## Goal-Driven Training Result\n"
                        f"Goal: {_metric_label} ≥ {_tgt_str} on '{_target_col}'\n"
                        f"Achieved: {'Yes' if goal_train_event['achieved'] else 'No'}\n"
                        f"Best algorithm: {goal_train_event['winner_algorithm_name']} "
                        f"({_metric_label} = {goal_train_event['winner_score']:.3f})\n"
                        "Trials: "
                        + ", ".join(
                            t["algorithm_name"] + f" ({t['score']:.3f})"
                            for t in goal_train_event["trials"]
                        )
                        + "\n"
                        "Summarise the result briefly: was the goal met, which algorithm "
                        "performed best, and what should the user do next (e.g. train that "
                        "algorithm fully, upload more data, or adjust the target)."
                    )
                except Exception:  # noqa: BLE001
                    pass  # Nice-to-have; never crash chat

    # Check if this is an auto-retrain status/toggle request
    conv_export_event: dict | None = None
    if _CONV_EXPORT_PATTERNS.search(body.message) and project:
        try:
            _msg_count = 0
            _conv_stmt = select(Conversation).where(
                Conversation.project_id == body.project_id
            )
            _conv = session.exec(_conv_stmt).first()
            if _conv:
                _msgs = json.loads(_conv.messages)
                _msg_count = len([m for m in _msgs if m.get("role") == "assistant"])
            conv_export_event = {
                "project_id": body.project_id,
                "download_url": f"/api/chat/{body.project_id}/export",
                "message_count": _msg_count,
                "dataset_name": ctx["dataset"].filename if ctx["dataset"] else None,
            }
            system_prompt += (
                "\n\n## Conversation Export\n"
                f"The user wants to export/download this analysis as an HTML report. "
                f"The export is ready and contains {_msg_count} messages. "
                "Tell them their analysis report is ready to download. "
                "Keep it brief — the download button will appear automatically."
            )
        except Exception:  # noqa: BLE001
            pass

    auto_retrain_event: dict | None = None
    if _AUTO_RETRAIN_PATTERNS.search(body.message) and ctx["project"]:
        try:
            from db import get_session as _gs

            _project = ctx["project"]
            _enabled = _project.auto_retrain

            # Detect enable/disable intent
            _msg_lower = body.message.lower()
            _enable_words = {"enable", "turn on", "activate", "start", "keep"}
            _disable_words = {"disable", "turn off", "deactivate", "stop"}
            _want_enable = any(w in _msg_lower for w in _enable_words)
            _want_disable = any(w in _msg_lower for w in _disable_words)

            if _want_enable or _want_disable:
                _new_state = _want_enable and not _want_disable
                with next(_gs()) as _s:
                    from models.project import Project as _Proj

                    _p = _s.get(_Proj, body.project_id)
                    if _p:
                        _p.auto_retrain = _new_state
                        _s.add(_p)
                        _s.commit()
                        _enabled = _new_state

            # Find selected model for display
            _sel_run_algo = None
            if ctx["model_runs"]:
                _sel_run = next(
                    (
                        mr
                        for mr in ctx["model_runs"]
                        if mr.is_selected and mr.status == "done"
                    ),
                    None,
                )
                _sel_run_algo = _sel_run.algorithm if _sel_run else None

            auto_retrain_event = {
                "project_id": body.project_id,
                "enabled": _enabled,
                "selected_algorithm": _sel_run_algo,
                "has_selected_model": _sel_run_algo is not None,
            }

            _status = "enabled" if _enabled else "disabled"
            _algo_msg = (
                f" Will use **{_sel_run_algo}** algorithm."
                if _sel_run_algo and _enabled
                else ""
            )
            system_prompt += (
                f"\n\n## Auto-Retrain Status\n"
                f"Auto-retrain is currently **{_status}**.\n"
                f"{_algo_msg}\n"
                "Tell the user the current auto-retrain status. If enabled, explain that the model "
                "will automatically retrain whenever new data is uploaded. If disabled, explain how "
                "to enable it or that they can ask you to turn it on. Keep it brief and friendly."
            )
        except Exception:  # noqa: BLE001
            pass  # Nice-to-have; never crash chat

    # Check if this is a prediction opportunity discovery request
    predict_opp_event: dict | None = None
    if _PREDICT_OPP_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.analyzer import analyze_dataframe as _adf
            from core.analyzer import compute_prediction_opportunities as _cpo
            from pathlib import Path as _Path

            _ds = ctx["dataset"]
            _fpath = _Path(_ds.file_path)
            if _fpath.exists():
                import pandas as _pd2

                _df2 = _pd2.read_csv(_fpath)
                _profile2 = _adf(_df2)
                _opps = _cpo(
                    col_stats=_profile2["columns"],
                    row_count=_profile2["row_count"],
                )
                predict_opp_event = {
                    "dataset_id": _ds.id,
                    "opportunities": _opps,
                    "total": len(_opps),
                }
                if _opps:
                    _top = _opps[0]
                    _top_col = _top["target_col"]
                    _top_type = _top["problem_type"]
                    _top_score = _top["feasibility_score"]
                    system_prompt += (
                        f"\n\n## Prediction Opportunities\n"
                        f"Top suggestion: predict **{_top_col}** ({_top_type}, "
                        f"feasibility {_top_score}/100). Total {len(_opps)} opportunities found.\n"
                        f"Opportunities: {', '.join(o['target_col'] for o in _opps)}.\n"
                        "Walk the analyst through these prediction opportunities in plain English. "
                        "Explain the top suggestion and why it would be valuable. "
                        "Mention they can click a card option to set any column as their prediction target. "
                        "Keep it conversational and encouraging."
                    )
        except Exception:  # noqa: BLE001
            pass  # Nice-to-have; never crash chat

    # Check if this is a dataset distribution comparison request
    dataset_compare_event: dict | None = None
    if _DATASET_COMPARE_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.analyzer import compute_dataset_comparison as _cdc
            from pathlib import Path as _PathC

            import pandas as _pdc

            # Find the two most recent datasets for this project
            from models.dataset import Dataset as _Dataset

            _all_ds = (
                session.query(_Dataset)
                .filter(_Dataset.project_id == body.project_id)
                .order_by(_Dataset.created_at.asc())
                .all()
            )
            if len(_all_ds) >= 2:
                _baseline_ds = _all_ds[0]
                _new_ds = _all_ds[-1]
                _bp = _PathC(_baseline_ds.file_path)
                _np = _PathC(_new_ds.file_path)
                if _bp.exists() and _np.exists():
                    _old_df = _pdc.read_csv(_bp)
                    _new_df_c = _pdc.read_csv(_np)
                    _drift = _cdc(_old_df, _new_df_c)
                    dataset_compare_event = {
                        "baseline_id": _baseline_ds.id,
                        "new_id": _new_ds.id,
                        "baseline_name": _baseline_ds.filename,
                        "new_name": _new_ds.filename,
                        **_drift,
                    }
                    _score = _drift["drift_score"]
                    _summary_c = _drift["summary"]
                    _n_numeric = len(_drift["numeric_drifts"])
                    _n_cat = len(_drift["categorical_drifts"])
                    system_prompt += (
                        f"\n\n## Dataset Comparison\n"
                        f"Comparing '{_baseline_ds.filename}' (baseline) vs '{_new_ds.filename}' (new).\n"
                        f"Overall drift score: {_score}/100. {_summary_c}\n"
                        f"Numeric shifts: {_n_numeric} columns. Categorical changes: {_n_cat} columns.\n"
                        f"New columns: {_drift['new_columns']}. Dropped columns: {_drift['dropped_columns']}.\n"
                        "Explain the distribution comparison to the analyst in plain English. "
                        "Highlight which columns changed most and what it means for model predictions. "
                        "Advise whether retraining is recommended based on the drift score."
                    )
        except Exception:  # noqa: BLE001
            pass  # Nice-to-have; never crash chat

    # Check for data version history request
    version_history_event: dict | None = None
    if _VERSION_HISTORY_PATTERNS.search(body.message):
        try:
            from pathlib import Path as _PathV

            import pandas as _pdv

            from core.analyzer import compute_version_history as _cvh
            from models.dataset import Dataset as _DatasetV

            _all_ds_v = (
                session.query(_DatasetV)
                .filter(_DatasetV.project_id == body.project_id)
                .order_by(_DatasetV.uploaded_at.asc())
                .all()
            )
            _ds_dicts = [
                {
                    "id": ds.id,
                    "filename": ds.filename,
                    "row_count": ds.row_count,
                    "column_count": ds.column_count,
                    "uploaded_at": ds.uploaded_at.isoformat() if ds.uploaded_at else "",
                    "size_bytes": ds.size_bytes,
                }
                for ds in _all_ds_v
            ]
            _dfs_v = []
            for ds in _all_ds_v:
                _p = _PathV(ds.file_path)
                _dfs_v.append(_pdv.read_csv(_p) if _p.exists() else _pdv.DataFrame())

            _vh = _cvh(_ds_dicts, _dfs_v)
            version_history_event = _vh
            _n_versions = _vh["version_count"]
            _stability = _vh["overall_stability"]
            _vh_summary = _vh["summary"]
            system_prompt += (
                f"\n\n## Data Version History\n"
                f"{_n_versions} dataset version{'s' if _n_versions != 1 else ''} on record. "
                f"Overall stability: {_stability}. {_vh_summary}\n"
                "Present the upload timeline to the analyst in plain English. "
                "For each version transition, mention how much the data changed and whether retraining is advised."
            )
        except Exception:  # noqa: BLE001
            pass  # Nice-to-have; never crash chat

    # Check if this is a model health / project health summary request
    health_summary_event: dict | None = None
    if _HEALTH_SUMMARY_PATTERNS.search(body.message):
        try:
            from core.analyzer import compute_project_health_summary as _chs
            from models.deployment import Deployment as _Dep

            _deployments = list(
                session.exec(
                    select(_Dep).where(
                        _Dep.project_id == body.project_id,
                        _Dep.is_active == True,  # noqa: E712
                    )
                ).all()
            )
            _now = datetime.now(UTC).replace(tzinfo=None)
            _dep_dicts = [
                {
                    "deployment_id": d.id,
                    "algorithm": d.algorithm,
                    "target_column": d.target_column,
                    "created_at": d.created_at,
                    "request_count": d.request_count,
                    "last_predicted_at": d.last_predicted_at,
                    "environment": d.environment,
                }
                for d in _deployments
            ]
            health_summary_event = _chs(_dep_dicts, now=_now)
            health_summary_event["project_id"] = body.project_id

            _n_alerts = len(health_summary_event["alerts"])
            _overall = health_summary_event["overall_status"]
            _summary_text = health_summary_event["summary"]
            system_prompt += (
                f"\n\n## Project Model Health\n"
                f"Overall status: **{_overall}**. {_summary_text}\n"
                + (
                    f"There {'are' if _n_alerts > 1 else 'is'} {_n_alerts} deployment{'s' if _n_alerts > 1 else ''} "
                    f"needing attention. Top issues: "
                    + "; ".join(
                        f"{a['name']}: {a['top_issue']}"
                        for a in health_summary_event["alerts"]
                        if a.get("top_issue")
                    )
                    if _n_alerts > 0
                    else "All deployed models look healthy."
                )
                + "\nTell the user their model health status in plain English. "
                "If there are issues, name each affected model and suggest the most important action. "
                "If everything is healthy, reassure them."
            )
        except Exception:  # noqa: BLE001
            pass  # Nice-to-have; never crash chat

    # Check if this is a tune/optimize request
    tune_data: dict | None = None
    if _TUNE_PATTERNS.search(body.message):
        completed_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
        selected_run = next((mr for mr in completed_runs if mr.is_selected), None)
        target_run = selected_run or (completed_runs[-1] if completed_runs else None)
        if target_run:
            from core.tuner import is_tunable as _is_tunable

            if _is_tunable(target_run.algorithm):
                tune_data = {
                    "model_run_id": target_run.id,
                    "algorithm": target_run.algorithm,
                    "metrics": (
                        json.loads(target_run.metrics) if target_run.metrics else {}
                    ),
                }
                system_prompt += (
                    f"\n\n## Hyperparameter Tuning Available\n"
                    f"The user is asking about improving model performance. "
                    f"Their current best model is {target_run.algorithm} "
                    f"(metrics: {tune_data['metrics']}). "
                    "Inform them that you can automatically tune the hyperparameters using "
                    "RandomizedSearchCV to find better settings — no technical knowledge needed. "
                    "Tell them the Tune button is now available in the Models tab, or they can "
                    "say 'go ahead and tune it' to start immediately."
                )

    # Check if this is a model health / retraining question
    health_data: dict | None = None
    if _HEALTH_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            deployment = ctx["deployment"]
            run_for_health = next(
                (mr for mr in ctx["model_runs"] if mr.id == deployment.model_run_id),
                None,
            )
            if run_for_health:
                from models.feedback_record import FeedbackRecord

                fb_records = list(
                    session.exec(
                        select(FeedbackRecord).where(
                            FeedbackRecord.deployment_id == deployment.id
                        )
                    ).all()
                )
                logs_for_health = list(
                    session.exec(
                        select(PredictionLog).where(
                            PredictionLog.deployment_id == deployment.id
                        )
                    ).all()
                )
                health_data = _compute_health(
                    deployment, run_for_health, fb_records, logs_for_health
                )
                score = health_data["health_score"]
                health_status = health_data["status"]
                system_prompt += (
                    f"\n\n## Model Health Check (just computed)\n"
                    f"Algorithm: {deployment.algorithm} | Health score: {score}/100 | "
                    f"Status: {health_status.upper()} | Age: {health_data['model_age_days']} day(s)\n"
                    f"Has feedback data: {health_data['has_feedback_data']} | "
                    f"Has drift data: {health_data['has_drift_data']}\n"
                    "Reference this health check in your response. Explain what the score means "
                    "and whether the user should consider retraining. If the model is healthy, "
                    "reassure them. If it's warning or critical, guide them to retrain using the "
                    "'Retrain' button in the Models tab or by clicking the health card."
                )
        except Exception:  # noqa: BLE001
            pass  # Health check is nice-to-have; never crash chat

    # Check if this is a drift-related question
    drift_data: dict | None = None
    if _DRIFT_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            deployment = ctx["deployment"]
            logs = list(
                session.exec(
                    select(PredictionLog).where(
                        PredictionLog.deployment_id == deployment.id
                    )
                ).all()
            )
            drift_data = _compute_drift(deployment, logs)
            system_prompt += (
                f"\n\n## Prediction Drift Check (just computed)\n"
                f"Status: {drift_data['status']} | "
                f"Drift score: {drift_data['drift_score'] if drift_data['drift_score'] is not None else 'N/A'}/100\n"
                f"{drift_data['explanation']}\n"
                "Reference this drift analysis in your response. Help the user understand "
                "what drift means and whether they need to take action."
            )
        except Exception:  # noqa: BLE001
            pass  # Drift check is nice-to-have; never crash chat

    # Check for cross-deployment alerts request
    alerts_data: dict | None = None
    if _ALERTS_PATTERNS.search(body.message):
        try:
            active_deployments = list(
                session.exec(
                    select(Deployment).where(
                        Deployment.project_id == project_id,
                        Deployment.is_active == True,  # noqa: E712
                    )
                ).all()
            )
            alert_list: list[dict] = []
            now_ts = datetime.now(UTC).replace(tzinfo=None)

            for dep in active_deployments:
                run_a = session.get(ModelRun, dep.model_run_id)
                alg = dep.algorithm or "model"
                age_d = 0
                if run_a and run_a.created_at:
                    age_d = max(0, (now_ts - run_a.created_at).days)
                if age_d > 60:
                    alert_list.append(
                        {
                            "deployment_id": dep.id,
                            "algorithm": alg,
                            "severity": "critical" if age_d > 90 else "warning",
                            "type": "stale_model",
                            "message": f"'{alg}' is {age_d} days old.",
                            "recommendation": "Consider retraining with more recent data.",
                        }
                    )
                if dep.request_count == 0 and dep.created_at:
                    dep_age = max(0, (now_ts - dep.created_at).days)
                    if dep_age >= 1:
                        alert_list.append(
                            {
                                "deployment_id": dep.id,
                                "algorithm": alg,
                                "severity": "warning",
                                "type": "no_predictions",
                                "message": f"'{alg}' has been deployed {dep_age} day(s) with no predictions.",
                                "recommendation": "Share the dashboard link to start receiving predictions.",
                            }
                        )

            alerts_data = {
                "project_id": project_id,
                "alert_count": len(alert_list),
                "critical_count": sum(
                    1 for a in alert_list if a["severity"] == "critical"
                ),
                "warning_count": sum(
                    1 for a in alert_list if a["severity"] == "warning"
                ),
                "alerts": alert_list,
            }
            alert_summary = (
                f"{len(alert_list)} alert(s) found: "
                f"{alerts_data['critical_count']} critical, {alerts_data['warning_count']} warning."
                if alert_list
                else "No active alerts — all deployments look healthy."
            )
            system_prompt += (
                f"\n\n## Deployment Alerts (just scanned)\n{alert_summary}\n"
                "Summarise the alert status for the user. If there are critical alerts, "
                "guide them on what to do. If everything is healthy, reassure them."
            )
        except Exception:  # noqa: BLE001
            pass  # Alerts are nice-to-have; never crash chat

    # Check for model version history request
    history_event: dict | None = None
    if _HISTORY_PATTERNS.search(body.message) and ctx["model_runs"]:
        completed = [mr for mr in ctx["model_runs"] if mr.status == "done"]
        if len(completed) >= 2:
            history_event = {"project_id": project_id}
            system_prompt += (
                "\n\n## Model Version History\n"
                f"The project has {len(completed)} completed training run(s). "
                "The Version History card is now visible in the Models tab — it shows "
                "a timeline of model performance and trend direction. "
                "Tell the user their model history is available in the Models tab."
            )

    # Check for prediction analytics request
    analytics_event: dict | None = None
    if _ANALYTICS_PATTERNS.search(body.message) and ctx["deployment"]:
        dep_for_analytics = ctx["deployment"]
        count = dep_for_analytics.request_count
        analytics_event = {
            "deployment_id": dep_for_analytics.id,
            "total_predictions": count,
        }
        system_prompt += (
            f"\n\n## Prediction Analytics\n"
            f"The active deployment has logged {count} prediction(s) total. "
            "The Analytics card is visible in the Deployment tab with a usage chart. "
            "Reference the prediction count in your response and mention the Analytics card."
        )

    # Check for data cleaning suggestion request
    # Vision: "Explain before executing" — we suggest the operation, user confirms via button.
    cleaning_suggestion: dict | None = None
    if _CLEAN_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _cols = list(_df.columns)
                _op = _detect_clean_op(body.message, _cols)
                # Build a quality summary for context
                _null_counts = {
                    col: int(_df[col].isna().sum())
                    for col in _cols
                    if _df[col].isna().any()
                }
                _dup_count = int(_df.duplicated().sum())
                _context_parts = []
                if _dup_count > 0:
                    _context_parts.append(f"{_dup_count} duplicate row(s)")
                if _null_counts:
                    _top = sorted(_null_counts.items(), key=lambda x: -x[1])[:3]
                    _context_parts.append(
                        "missing values in: "
                        + ", ".join(f"'{k}' ({v})" for k, v in _top)
                    )
                cleaning_suggestion = {
                    "dataset_id": _ds.id,
                    "suggested_operation": _op,
                    "quality_summary": {
                        "duplicate_rows": _dup_count,
                        "missing_value_columns": _null_counts,
                        "total_rows": len(_df),
                    },
                }
                _ctx_text = (
                    "; ".join(_context_parts)
                    if _context_parts
                    else "no obvious issues detected"
                )
                system_prompt += (
                    f"\n\n## Data Cleaning Context\n"
                    f"Dataset quality: {_ctx_text}. "
                    + (
                        f"The user seems to want: {_op['operation'].replace('_', ' ')} "
                        + (
                            f"on column '{_op.get('column')}'"
                            if _op and _op.get("column")
                            else ""
                        )
                        + ". A cleaning suggestion card is shown — explain what it will do and ask the user to confirm before applying."
                        if _op
                        else "Describe the available cleaning operations (remove duplicates, fill missing values, filter rows, cap outliers, drop columns) and let the user choose."
                    )
                )
        except Exception:  # noqa: BLE001
            pass  # Cleaning suggestion is nice-to-have; never crash chat

    # Check for cross-tabulation / pivot table request
    crosstab_event: dict | None = None
    if _CROSSTAB_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _cols = (
                    [c["name"] for c in json.loads(_ds.columns)]
                    if _ds.columns
                    else list(_df.columns)
                )
                _crosstab_req = _detect_crosstab_request(body.message, _cols)
                if _crosstab_req:
                    from core.chart_builder import build_crosstab as _build_crosstab

                    _ct_result = _build_crosstab(
                        _df,
                        row_col=_crosstab_req["row_col"],
                        col_col=_crosstab_req["col_col"],
                        value_col=_crosstab_req.get("value_col"),
                    )
                    crosstab_event = _ct_result
                    system_prompt += (
                        f"\n\n## Pivot Table\n"
                        f"{_ct_result['summary']}\n"
                        "A pivot table has been generated and is shown inline in the chat. "
                        "Tell the user what you see — highlight the highest and lowest values, "
                        "and suggest what patterns are worth investigating further."
                    )
        except Exception:  # noqa: BLE001
            pass  # Crosstab is nice-to-have; never crash chat

    # Check for anomaly detection request
    anomaly_event: dict | None = None
    if _ANOMALY_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.anomaly import detect_anomalies as _detect

            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _numeric_cols = _df.select_dtypes(include="number").columns.tolist()[
                    :10
                ]
                if _numeric_cols:
                    _result = _detect(
                        _df, features=_numeric_cols, contamination=0.05, n_top=10
                    )
                    anomaly_event = {"dataset_id": _ds.id, **_result}
                    system_prompt += (
                        f"\n\n## Anomaly Detection Results\n"
                        f"{_result['summary']}\n"
                        f"Features analysed: {', '.join(_result['features_used'])}.\n"
                        "The Anomaly Detection card is now visible in the Data tab. "
                        "Tell the user what you found and suggest they examine the top anomalous rows."
                    )
        except Exception:  # noqa: BLE001
            pass  # Anomaly detection is nice-to-have; never crash chat

    # Check for computed column request ("add column margin = revenue / cost")
    compute_suggestion: dict | None = None
    if _COMPUTE_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _cols = list(_df.columns)
                _compute_req = _detect_compute_request(body.message, _cols)
                if _compute_req:
                    from core.computed import preview_computed_column as _preview_col

                    _preview = _preview_col(
                        _df, _compute_req["name"], _compute_req["expression"]
                    )
                    compute_suggestion = {
                        "dataset_id": _ds.id,
                        "name": _compute_req["name"],
                        "expression": _compute_req["expression"],
                        "sample_values": _preview["sample_values"],
                        "dtype": _preview["dtype"],
                    }
                    system_prompt += (
                        f"\n\n## Computed Column Suggestion\n"
                        f"The user wants to add a new column '{_compute_req['name']}' "
                        f"= {_compute_req['expression']}. "
                        f"Sample values: {_preview['sample_values'][:3]}. "
                        "A Compute Card is shown in the Data tab. "
                        "Confirm the column looks correct and ask the user to click Apply to add it."
                    )
        except Exception:  # noqa: BLE001
            pass  # Compute suggestion is nice-to-have; never crash chat

    # Check for segment comparison request ("compare enterprise vs SMB")
    segment_comparison_event: dict | None = None
    if _COMPARE_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _compare_req = _detect_compare_request(body.message, _df)
                if _compare_req:
                    from core.analyzer import compare_segments as _compare_segs

                    _seg_result = _compare_segs(
                        _df,
                        _compare_req["group_col"],
                        _compare_req["val1"],
                        _compare_req["val2"],
                    )
                    segment_comparison_event = _seg_result
                    system_prompt += (
                        f"\n\n## Segment Comparison\n"
                        f"{_seg_result['summary']}\n"
                        f"Groups: '{_compare_req['val1']}' ({_seg_result['count1']} rows) "
                        f"vs '{_compare_req['val2']}' ({_seg_result['count2']} rows) "
                        f"in column '{_compare_req['group_col']}'.\n"
                    )
                    if _seg_result["notable_diffs"]:
                        _top = _seg_result["notable_diffs"][:3]
                        _diff_lines = []
                        for _nd in _top:
                            _es = round(abs(_nd["effect_size"]), 2)
                            _dir = (
                                f"higher in '{_compare_req['val1']}'"
                                if _nd["direction"] == "higher_in_val1"
                                else f"higher in '{_compare_req['val2']}'"
                            )
                            _diff_lines.append(
                                f"- {_nd['name']}: {_dir} (effect={_es})"
                            )
                        system_prompt += (
                            "Notable differences:\n" + "\n".join(_diff_lines) + "\n"
                        )
                    system_prompt += (
                        "A Segment Comparison table is shown in the chat. "
                        "Narrate the key business insights from these differences."
                    )
        except Exception:  # noqa: BLE001
            pass  # Comparison is nice-to-have; never crash chat

    # Check if user is asking about data readiness / quality before training
    data_readiness_event: dict | None = None
    if _DATA_READINESS_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                from core.readiness import compute_data_readiness as _compute_dr

                _fs = ctx["feature_set"]
                _target = _fs.target_column if _fs else None
                _dr_result = _compute_dr(_df, target_col=_target)
                data_readiness_event = {"dataset_id": _ds.id, **_dr_result}
                _score = _dr_result["score"]
                _grade = _dr_result["grade"]
                _summary = _dr_result["summary"]
                _recs = _dr_result["recommendations"]
                _rec_text = " Recommendations: " + "; ".join(_recs[:3]) if _recs else ""
                system_prompt += (
                    f"\n\n## Data Readiness Check\n"
                    f"Score: {_score}/100 (Grade {_grade}) — {_dr_result['status'].replace('_', ' ').title()}.\n"
                    f"{_summary}{_rec_text}\n"
                    "A data readiness card is shown in the chat. "
                    "Narrate the key findings and recommend what the user should do next."
                )
        except Exception:  # noqa: BLE001
            pass  # Data readiness is nice-to-have; never crash chat

    # Check for target correlation request ("what drives revenue?", "correlated with profit?")
    target_correlation_event: dict | None = None
    if _CORRELATION_TARGET_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _all_cols = _df.columns.tolist()
                _target_col = _detect_correlation_target_request(
                    body.message, _all_cols
                )
                # Fall back to the feature-set target if the user said "what drives X"
                # but didn't name a column exactly
                if (
                    not _target_col
                    and ctx["feature_set"]
                    and ctx["feature_set"].target_column
                ):
                    _target_col = ctx["feature_set"].target_column
                if _target_col:
                    from core.analyzer import analyze_target_correlations as _atc

                    _corr_result = _atc(_df, _target_col)
                    if not _corr_result.get("error"):
                        target_correlation_event = {
                            "dataset_id": _ds.id,
                            **_corr_result,
                        }
                        _top = _corr_result["correlations"][:3]
                        _top_desc = ", ".join(
                            f"{e['column']} (r={e['correlation']:+.2f})" for e in _top
                        )
                        system_prompt += (
                            f"\n\n## Target Correlation Analysis\n"
                            f"Analysed correlations with '{_target_col}'.\n"
                            f"{_corr_result['summary']}\n"
                            f"Top correlates: {_top_desc}.\n"
                            "A correlation chart is shown in the chat. "
                            "Narrate which factors matter most and what this means for modeling."
                        )
        except Exception:  # noqa: BLE001
            pass  # Correlation is nice-to-have; never crash chat

    # Check for time-series forecast request ("predict next 3 months", "forecast sales")
    forecast_event: dict | None = None
    if _FORECAST_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                from core.forecaster import detect_time_series as _detect_ts
                from core.forecaster import forecast_next_periods as _forecast

                _ts_info = _detect_ts(_df)
                if _ts_info:
                    _fc_params = _detect_forecast_request(body.message)
                    _value_col = _ts_info["value_cols"][0]
                    _fc_result = _forecast(
                        _df,
                        _ts_info["date_col"],
                        _value_col,
                        periods=_fc_params["periods"],
                    )
                    forecast_event = _fc_result
                    system_prompt += (
                        f"\n\n## Time-Series Forecast\n"
                        f"{_fc_result['summary']}\n"
                        f"Column forecasted: {_value_col}. "
                        f"Trend: {_fc_result['trend']} "
                        f"({_fc_result['growth_pct']:+.1f}% over forecast horizon).\n"
                        "A forecast chart is shown in the chat. "
                        "Narrate the key business insights and what the trend means."
                    )
        except Exception:  # noqa: BLE001
            pass  # Forecast is nice-to-have; never crash chat

    # Check for group-by analysis ("revenue by region", "breakdown by product", etc.)
    group_stats_event: dict | None = None
    if _GROUP_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _grp_req = _detect_group_request(body.message, _df)
                if _grp_req:
                    from core.analyzer import compute_group_stats as _cgs

                    _grp_result = _cgs(
                        _df,
                        _grp_req["group_col"],
                        value_cols=_grp_req.get("value_cols"),
                        agg=_grp_req.get("agg", "sum"),
                    )
                    if not _grp_result.get("error"):
                        group_stats_event = {
                            "dataset_id": _ds.id,
                            **_grp_result,
                        }
                        system_prompt += (
                            f"\n\n## Group-By Analysis\n"
                            f"{_grp_result['summary']}\n"
                            f"Top groups by {_grp_result['value_col']} ({_grp_result['agg']}): "
                            + ", ".join(
                                str(r.get("group", "?"))
                                for r in _grp_result["rows"][:5]
                            )
                            + ".\n"
                            "A grouped bar chart is shown in the chat. "
                            "Narrate the key business insights: which group leads, gaps, surprises."
                        )
        except Exception:  # noqa: BLE001
            pass  # Group stats are nice-to-have; never crash chat

    # Check for full correlation matrix / heatmap request
    heatmap_chart: dict | None = None
    if _HEATMAP_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                import json as _json_hm

                # Try cached profile first
                _correlations: dict = {}
                if _ds.profile:
                    try:
                        _prof = _json_hm.loads(_ds.profile)
                        _correlations = _prof.get("correlations", {})
                    except Exception:  # noqa: BLE001
                        pass
                if not _correlations:
                    _df = _load_working_df(_file_path, _active_filter_conditions)
                    from core.analyzer import compute_full_profile as _cfp

                    _correlations = _cfp(_df).get("correlations", {})
                _cols = _correlations.get("columns", [])
                _matrix = _correlations.get("matrix", [])
                if len(_cols) >= 2:
                    from core.chart_builder import (
                        build_correlation_heatmap as _build_hm,
                    )

                    heatmap_chart = _build_hm(_matrix, _cols)
                    system_prompt += (
                        f"\n\n## Correlation Matrix\n"
                        f"Full pairwise Pearson correlation matrix for {len(_cols)} numeric columns: "
                        f"{', '.join(_cols[:6])}{'...' if len(_cols) > 6 else ''}.\n"
                        "A heatmap is shown in the chat. Narrate the strongest and most surprising "
                        "correlations, positive and negative. Help the user understand what these "
                        "relationships mean for their data."
                    )
        except Exception:  # noqa: BLE001
            pass  # Heatmap is nice-to-have; never crash chat

    # Check for scatter plot request ("plot revenue vs quantity")
    scatter_chart: dict | None = None
    if _SCATTER_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _scatter_req = _detect_scatter_request(body.message, _df)
                if _scatter_req:
                    _x_col = _scatter_req["x_col"]
                    _y_col = _scatter_req["y_col"]
                    _x_vals = _df[_x_col].dropna().tolist()
                    _y_vals = _df[_y_col].dropna().tolist()
                    # Align lengths (common index after dropna per column)
                    _aligned = _df[[_x_col, _y_col]].dropna()
                    _x_vals = _aligned[_x_col].tolist()
                    _y_vals = _aligned[_y_col].tolist()
                    # Cap at 500 points for rendering performance
                    if len(_x_vals) > 500:
                        import random as _random

                        _idx = sorted(_random.sample(range(len(_x_vals)), 500))
                        _x_vals = [_x_vals[i] for i in _idx]
                        _y_vals = [_y_vals[i] for i in _idx]
                    from core.chart_builder import build_scatter_chart as _build_sc

                    scatter_chart = _build_sc(
                        _x_vals,
                        _y_vals,
                        title=f"{_x_col} vs {_y_col}",
                        x_label=_x_col,
                        y_label=_y_col,
                    )
                    # Compute Pearson r for system prompt context
                    import numpy as _np

                    _r: float | None = None
                    if len(_x_vals) >= 3:
                        try:
                            _r = float(_np.corrcoef(_x_vals, _y_vals)[0, 1])
                        except Exception:  # noqa: BLE001
                            pass
                    _r_text = (
                        f"Pearson r = {_r:.3f} ({'positive' if _r > 0 else 'negative'} correlation, "
                        f"{'strong' if abs(_r) > 0.7 else 'moderate' if abs(_r) > 0.4 else 'weak'})"
                        if _r is not None
                        else "correlation could not be computed"
                    )
                    system_prompt += (
                        f"\n\n## Scatter Plot: {_x_col} vs {_y_col}\n"
                        f"Showing {len(_x_vals)} data points. {_r_text}.\n"
                        "A scatter plot is shown in the chat. Describe the relationship: "
                        "direction, strength, any clusters or outliers visible, and what "
                        "this means for the analyst's data."
                    )
        except Exception:  # noqa: BLE001
            pass  # Scatter chart is nice-to-have; never crash chat

    # Check for line/trend chart request ("plot revenue over time", "trend of sales")
    line_chart: dict | None = None
    if _LINE_CHART_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _line_req = _detect_line_chart_request(body.message, _df)
                if _line_req:
                    _val_cols = _line_req["value_cols"]
                    _date_col = _line_req["date_col"]
                    import pandas as _pd_lc

                    _df[_date_col] = _pd_lc.to_datetime(_df[_date_col], errors="coerce")
                    _df_ts = _df.dropna(subset=[_date_col]).sort_values(_date_col)
                    _dates = _df_ts[_date_col].astype(str).tolist()
                    # Cap at 500 points for rendering
                    if len(_dates) > 500:
                        _step = len(_dates) // 500
                        _dates = _dates[::_step]
                        _df_ts = _df_ts.iloc[::_step].reset_index(drop=True)

                    if len(_val_cols) == 1:
                        # Single column: enrich with rolling avg + OLS trend line
                        _val_col = _val_cols[0]
                        _vals = _df_ts[_val_col].tolist()
                        from core.chart_builder import (
                            build_timeseries_chart as _build_ts,
                        )

                        line_chart = _build_ts(_dates, _vals, _val_col)
                        _first_val = next((v for v in _vals if v is not None), None)
                        _last_val = next(
                            (v for v in reversed(_vals) if v is not None), None
                        )
                        _trend_text = ""
                        if (
                            _first_val is not None
                            and _last_val is not None
                            and _first_val != 0
                        ):
                            _pct = (_last_val - _first_val) / abs(_first_val) * 100
                            _trend_text = (
                                f"Overall trend: {'up' if _pct > 0 else 'down'} {abs(_pct):.1f}% "
                                f"(from {_first_val:.2f} to {_last_val:.2f})."
                            )
                        system_prompt += (
                            f"\n\n## Trend Chart: {_val_col} over time\n"
                            f"Date column: {_date_col}. {len(_dates)} data points. {_trend_text}\n"
                            "A line chart is shown in the chat with raw values, rolling average, "
                            "and trend line. Describe the overall direction, any notable peaks or "
                            "dips, and what this means for the analyst."
                        )
                    else:
                        # Multiple columns: overlay chart with raw series per column
                        from core.chart_builder import (
                            build_overlay_chart as _build_overlay,
                        )

                        _cols_data = {col: _df_ts[col].tolist() for col in _val_cols}
                        _overlay_title = f"{', '.join(_val_cols)} over time"
                        line_chart = _build_overlay(_dates, _cols_data, _overlay_title)
                        # Build per-column summary for the LLM prompt
                        _col_summaries = []
                        for _oc in _val_cols:
                            _ov = _df_ts[_oc].tolist()
                            _of = next((v for v in _ov if v is not None), None)
                            _ol = next(
                                (v for v in reversed(_ov) if v is not None), None
                            )
                            if _of is not None and _ol is not None and _of != 0:
                                _op = (_ol - _of) / abs(_of) * 100
                                _col_summaries.append(
                                    f"{_oc}: {'up' if _op > 0 else 'down'} {abs(_op):.1f}% "
                                    f"({_of:.2f}→{_ol:.2f})"
                                )
                        system_prompt += (
                            f"\n\n## Overlay Chart: {', '.join(_val_cols)} over time\n"
                            f"Date column: {_date_col}. {len(_dates)} data points. "
                            f"Per-metric trend: {'; '.join(_col_summaries)}.\n"
                            "An overlay line chart is shown in the chat with one line per metric. "
                            "Compare the trends across all metrics — describe which moved more, "
                            "whether they moved together or diverged, and what this means."
                        )
        except Exception:  # noqa: BLE001
            pass  # Line chart is nice-to-have; never crash chat

    # Check for box plot request ("distribution of revenue by region", "box plot of sales")
    boxplot_chart: dict | None = None
    if _BOXPLOT_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _bp_req = _detect_boxplot_request(body.message, _df)
                if _bp_req:
                    _bp_val = _bp_req["value_col"]
                    _bp_grp = _bp_req["group_col"]
                    from core.chart_builder import build_boxplot as _build_bp

                    boxplot_chart = _build_bp(_df, _bp_val, _bp_grp)
                    _bp_groups = (
                        f"grouped by {_bp_grp} ({_df[_bp_grp].nunique()} categories)"
                        if _bp_grp
                        else "overall distribution"
                    )
                    _bp_median = (
                        float(_df[_bp_val].median()) if not _df[_bp_val].empty else 0
                    )
                    system_prompt += (
                        f"\n\n## Box Plot: {_bp_val} ({_bp_groups})\n"
                        f"Median: {_bp_median:.2f}. "
                        "A box-and-whisker chart is shown in the chat. "
                        "Describe the spread, median, any outliers visible, "
                        f"and{' how the groups compare' if _bp_grp else ' what the distribution looks like'}."
                    )
        except Exception:  # noqa: BLE001
            pass  # Box plot is nice-to-have; never crash chat

    # Check for pie / donut chart request ("pie chart of revenue by region")
    pie_chart: dict | None = None
    if _PIE_CHART_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _pie_req = _detect_pie_chart_request(body.message, _df)
                if _pie_req:
                    _pie_val = _pie_req["value_col"]
                    _pie_slc = _pie_req["slice_col"]
                    from core.chart_builder import build_pie_chart as _build_pie

                    _pie_series = (
                        _df.groupby(_pie_slc)[_pie_val]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    pie_chart = _build_pie(
                        _pie_series,
                        title=f"{_pie_val} by {_pie_slc}",
                        limit=10,
                    )
                    _pie_total = (
                        float(_pie_series.sum()) if not _pie_series.empty else 0
                    )
                    _pie_top = (
                        f"{_pie_series.index[0]} ({_pie_series.iloc[0] / _pie_total * 100:.1f}%)"
                        if _pie_total > 0 and len(_pie_series) > 0
                        else "unknown"
                    )
                    system_prompt += (
                        f"\n\n## Pie Chart: {_pie_val} by {_pie_slc}\n"
                        f"Total: {_pie_total:.2f}. Largest slice: {_pie_top}. "
                        f"{_pie_series.shape[0]} categories shown. "
                        "A pie chart is shown in the chat. Describe the composition: "
                        "which slice dominates, how concentrated is the distribution, "
                        "any surprising or notable segments."
                    )
        except Exception:  # noqa: BLE001
            pass  # Pie chart is nice-to-have; never crash chat

    # Check for bar chart request ("bar chart of revenue by region")
    bar_chart: dict | None = None
    if _BAR_CHART_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _bar_req = _detect_bar_chart_request(body.message, _df)
                if _bar_req:
                    _bar_val = _bar_req["value_col"]
                    _bar_grp = _bar_req["group_col"]
                    _bar_agg = _bar_req["agg"]
                    from core.chart_builder import build_bar_chart as _build_bar

                    if _bar_grp:
                        _bar_series = (
                            _df.groupby(_bar_grp)[_bar_val]
                            .agg(_bar_agg)
                            .sort_values(ascending=False)
                        )
                        _bar_title = (
                            f"{_bar_agg.capitalize()} of {_bar_val} by {_bar_grp}"
                        )
                    else:
                        # No group column — bar chart of value column's values
                        _bar_series = _df[_bar_val].head(20)
                        _bar_title = f"{_bar_val} (first 20 rows)"
                    bar_chart = _build_bar(
                        _bar_series,
                        title=_bar_title,
                        x_label=_bar_grp or "",
                        y_label=_bar_val,
                        limit=20,
                    )
                    _bar_top = (
                        f"{_bar_series.index[0]}: {_bar_series.iloc[0]:.2f}"
                        if _bar_grp and not _bar_series.empty
                        else (
                            str(_bar_series.iloc[0]) if not _bar_series.empty else "n/a"
                        )
                    )
                    system_prompt += (
                        f"\n\n## Bar Chart: {_bar_title}\n"
                        f"Aggregation: {_bar_agg}. Top bar: {_bar_top}. "
                        f"{len(_bar_series)} groups shown. "
                        "A vertical bar chart is shown in the chat. Describe the key findings: "
                        "which group leads, any notable outliers or patterns."
                    )
        except Exception:  # noqa: BLE001
            pass  # Bar chart is nice-to-have; never crash chat

    # Check for dataset download/export request ("download my data", "export to CSV")
    data_export: dict | None = None
    if _DOWNLOAD_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df_export = _load_working_df(_file_path, _active_filter_conditions)
                _is_filtered = bool(_active_filter_conditions)
                _export_filename = (
                    Path(_ds.filename).stem
                    + ("_filtered" if _is_filtered else "")
                    + ".csv"
                )
                data_export = {
                    "dataset_id": _ds.id,
                    "filename": _export_filename,
                    "row_count": len(_df_export),
                    "filtered": _is_filtered,
                    "download_url": f"/api/data/{_ds.id}/download",
                }
                _filter_note = (
                    f" (filtered to {len(_df_export)} of {len(pd.read_csv(_file_path))} rows)"
                    if _is_filtered
                    else ""
                )
                system_prompt += (
                    f"\n\n## Dataset Export Ready\n"
                    f"File: {_export_filename}{_filter_note}. "
                    "The user can download the dataset as a CSV. "
                    "Tell them their data is ready to download via the export card."
                )
        except Exception:  # noqa: BLE001
            pass  # Export is nice-to-have; never crash chat

    # Check for histogram request ("histogram of revenue", "frequency histogram of age")
    histogram_chart: dict | None = None
    if _HISTOGRAM_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                import numpy as _np_hist

                _df = _load_working_df(_file_path, _active_filter_conditions)
                _hist_col = _detect_histogram_col(body.message, _df)
                if _hist_col:
                    _vals = _df[_hist_col].dropna().values
                    _n_bins = min(30, max(5, len(_vals) // 10))
                    _counts, _bin_edges = _np_hist.histogram(_vals, bins=_n_bins)
                    from core.chart_builder import build_histogram as _build_hist

                    histogram_chart = _build_hist(
                        bins=_bin_edges[:-1].tolist(),
                        counts=_counts.tolist(),
                        title=f"Distribution of {_hist_col}",
                        x_label=_hist_col,
                        y_label="Count",
                    )
                    _mean = float(_np_hist.mean(_vals))
                    _std = float(_np_hist.std(_vals))
                    system_prompt += (
                        f"\n\n## Histogram: {_hist_col}\n"
                        f"Column: {_hist_col} | {len(_vals)} values | "
                        f"Mean: {_mean:.2f} | Std dev: {_std:.2f} | Bins: {_n_bins}.\n"
                        "A frequency histogram is shown in the chat. "
                        "Describe the shape of the distribution — is it symmetric, skewed, "
                        "bimodal? Are there any gaps or outlier bins?"
                    )
        except Exception:  # noqa: BLE001
            pass  # Histogram is nice-to-have; never crash chat

    # Check for missing values overview ("which columns have missing data?", "show nulls")
    null_map_event: dict | None = None
    if _NULL_MAP_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _total_rows = len(_df)
                _col_nulls = []
                for _col in _df.columns:
                    _null_count = int(_df[_col].isna().sum())
                    _null_pct = (
                        round(_null_count / _total_rows * 100, 1) if _total_rows else 0
                    )
                    _col_nulls.append(
                        {
                            "column": _col,
                            "null_count": _null_count,
                            "null_pct": _null_pct,
                            "complete_pct": round(100 - _null_pct, 1),
                        }
                    )
                # Sort by most missing first
                _col_nulls.sort(key=lambda x: x["null_pct"], reverse=True)
                _cols_with_nulls = [c for c in _col_nulls if c["null_count"] > 0]
                _fully_complete = len(_col_nulls) - len(_cols_with_nulls)
                _overall_completeness = (
                    round(
                        sum(c["complete_pct"] for c in _col_nulls) / len(_col_nulls), 1
                    )
                    if _col_nulls
                    else 100.0
                )
                _summary = (
                    f"{len(_cols_with_nulls)} of {len(_col_nulls)} columns have missing values. "
                    f"Overall completeness: {_overall_completeness}%."
                    if _cols_with_nulls
                    else f"All {len(_col_nulls)} columns are fully complete — no missing values!"
                )
                null_map_event = {
                    "dataset_id": _ds.id,
                    "total_rows": _total_rows,
                    "total_columns": len(_col_nulls),
                    "columns_with_nulls": len(_cols_with_nulls),
                    "fully_complete_columns": _fully_complete,
                    "overall_completeness": _overall_completeness,
                    "columns": _col_nulls,
                    "summary": _summary,
                }
                system_prompt += (
                    f"\n\n## Missing Values Overview\n"
                    f"{_summary}\n"
                    + (
                        "Worst columns: "
                        + ", ".join(
                            f"{c['column']} ({c['null_pct']}% missing)"
                            for c in _cols_with_nulls[:3]
                        )
                        + ".\n"
                        if _cols_with_nulls
                        else ""
                    )
                    + "A missing-values card is shown in the chat. "
                    "Narrate the data completeness situation. "
                    "If there are missing values, suggest specific cleaning actions."
                )
        except Exception:  # noqa: BLE001
            pass  # Null map is nice-to-have; never crash chat

    # Check for summary statistics request ("summarize my data", "stats for all columns")
    summary_stats_event: dict | None = None
    if _SUMMARY_STATS_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.analyzer import compute_summary_stats as _compute_ss

            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                summary_stats_event = _compute_ss(_df)
                summary_stats_event["dataset_id"] = _ds.id
                _ns = len(summary_stats_event["numeric_stats"])
                _cs = len(summary_stats_event["categorical_stats"])
                system_prompt += (
                    f"\n\n## Dataset Summary Statistics\n"
                    f"{summary_stats_event['summary']}\n"
                    f"{_ns} numeric column(s), {_cs} categorical column(s). "
                    "A summary statistics table is shown in the chat. "
                    "Narrate the key highlights — range of values, any column "
                    "with many nulls, the most common categorical values."
                )
        except Exception:  # noqa: BLE001
            pass  # Summary stats is nice-to-have; never crash chat

    # Check for value counts request ("most common values in region", "frequency table for product")
    value_counts_event: dict | None = None
    if _VALUE_COUNT_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.analyzer import compute_value_counts as _compute_vc

            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _vc_col = _detect_value_counts_col(body.message, _df)
                if _vc_col:
                    value_counts_event = _compute_vc(_df, col=_vc_col, n=20)
                    value_counts_event["dataset_id"] = _ds.id
                    system_prompt += (
                        f"\n\n## Value Counts: {_vc_col}\n"
                        f"{value_counts_event['summary']}\n"
                        "A value-frequency table is shown in the chat. "
                        "Narrate the distribution — what the most common values are, "
                        "whether one category dominates, and any notable patterns."
                    )
        except Exception:  # noqa: BLE001
            pass  # Value counts is nice-to-have; never crash chat

    # Check for pair correlation request ("correlation between revenue and cost")
    pair_corr_event: dict | None = None
    if _PAIR_CORR_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.analyzer import compute_pair_correlation as _compute_pc

            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _pc_cols = _detect_pair_corr_cols(body.message, _df)
                if _pc_cols:
                    _col1, _col2 = _pc_cols
                    pair_corr_event = _compute_pc(_df, col1=_col1, col2=_col2)
                    pair_corr_event["dataset_id"] = _ds.id
                    system_prompt += (
                        f"\n\n## Pair Correlation: {_col1} vs {_col2}\n"
                        f"{pair_corr_event['summary']}\n"
                        "A PairCorrelationCard is shown in the chat with the exact r value "
                        "and significance. Narrate what this means for the analyst — is this "
                        "relationship meaningful? Should they use both columns in a model or is "
                        "one redundant? Suggest a follow-up question."
                    )
        except Exception:  # noqa: BLE001
            pass  # Pair correlation is nice-to-have; never crash chat

    # Check for stat query request ("what's the average revenue?", "total sales")
    stat_query_event: dict | None = None
    if _STAT_QUERY_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.analyzer import compute_stat_query as _compute_sq

            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _sq_params = _detect_stat_query(body.message, _df)
                if _sq_params:
                    stat_query_event = _compute_sq(
                        _df, agg=_sq_params["agg"], col=_sq_params["col"]
                    )
                    stat_query_event["dataset_id"] = _ds.id
                    system_prompt += (
                        f"\n\n## Stat Query Result\n"
                        f"{stat_query_event['summary']}\n"
                        "A StatQueryCard is shown in the chat with the computed value. "
                        "Narrate the result briefly — put it in context (is it high, low, "
                        "expected?) and suggest what the analyst might want to explore next."
                    )
        except Exception:  # noqa: BLE001
            pass  # Stat query is nice-to-have; never crash chat

    # Check for group trend request ("which regions are growing?", "fastest growing product?")
    group_trends_event: dict | None = None
    if _GROUP_TREND_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.analyzer import compute_group_trends as _compute_gt

            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _gt_params = _detect_group_trend_request(body.message, _df)
                if _gt_params:
                    _gt_result = _compute_gt(
                        _df,
                        date_col=_gt_params["date_col"],
                        group_col=_gt_params["group_col"],
                        value_col=_gt_params["value_col"],
                    )
                    if "error" not in _gt_result:
                        group_trends_event = _gt_result
                        group_trends_event["dataset_id"] = _ds.id
                        system_prompt += (
                            f"\n\n## Group Trend Analysis: {_gt_params['value_col']} by {_gt_params['group_col']} over time\n"
                            f"{_gt_result['summary']}\n"
                            f"Rising: {_gt_result['rising']} group(s), Falling: {_gt_result['falling']} group(s), "
                            f"Flat: {_gt_result['flat']} group(s).\n"
                            "A GroupTrendCard is shown in the chat ranking groups by growth rate. "
                            "Narrate the key findings — which groups are growing fastest, which are declining, "
                            "and what this might mean for the analyst's business decisions."
                        )
        except Exception:  # noqa: BLE001
            pass  # Group trends are nice-to-have; never crash chat

    # Check for column rename request ("rename revenue_usd to Revenue")
    rename_result: dict | None = None
    if _RENAME_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _cols = list(_df.columns)
                _rename_req = _detect_rename_request(body.message, _cols)
                if _rename_req:
                    _old = _rename_req["old_name"]
                    _new = _rename_req["new_name"]
                    # Validate new name
                    import re as _re_rename

                    if _re_rename.match(r"^\w+$", _new) and _new not in _cols:
                        _df = _df.rename(columns={_old: _new})
                        _df.to_csv(_file_path, index=False)
                        from core.analyzer import compute_full_profile as _cfp_rn

                        _profile_rn = _cfp_rn(_df)
                        with Session(session.bind) as _save_s:
                            _ds_rn = _save_s.get(Dataset, _ds.id)
                            if _ds_rn:
                                _ds_rn.profile = json.dumps(_profile_rn, default=str)
                                _ds_rn.columns = json.dumps(_profile_rn["columns"])
                                _ds_rn.column_count = len(_df.columns)
                                _save_s.add(_ds_rn)
                                _save_s.commit()
                        rename_result = {
                            "dataset_id": _ds.id,
                            "old_name": _old,
                            "new_name": _new,
                            "column_count": len(_df.columns),
                        }
                        system_prompt += (
                            f"\n\n## Column Renamed\n"
                            f"Column '{_old}' has been renamed to '{_new}'. "
                            "Confirm this to the user in a friendly way and note that the dataset "
                            "profile has been updated. If they had feature engineering or a model "
                            "trained on the old column name, remind them to check the Features tab."
                        )
        except Exception:  # noqa: BLE001
            pass  # Rename is nice-to-have; never crash chat

    # Check if user has new data to upload — guide them through the refresh workflow
    refresh_prompt_event: dict | None = None
    if _REFRESH_PATTERNS.search(body.message) and ctx["dataset"]:
        _ds = ctx["dataset"]
        _fs = ctx["feature_set"]
        _old_cols = [c["name"] for c in json.loads(_ds.columns)] if _ds.columns else []
        refresh_prompt_event = {
            "dataset_id": _ds.id,
            "current_filename": _ds.filename,
            "current_row_count": _ds.row_count,
            "required_columns": _old_cols,
        }
        _feature_note = ""
        if _fs and _fs.column_mapping:
            _req = list(json.loads(_fs.column_mapping).keys())
            _feature_note = (
                f" Your model uses these columns: {', '.join(_req[:5])}"
                + (" and more" if len(_req) > 5 else "")
                + "."
            )
        system_prompt += (
            f"\n\n## Data Refresh\n"
            f"The user wants to replace their current dataset ('{_ds.filename}', "
            f"{_ds.row_count} rows).{_feature_note} "
            "Tell them to drag their new file into the upload area (Data tab) or click the "
            "'Replace Data' button — the platform will check column compatibility and update "
            "the dataset in-place, preserving their model configuration. "
            "Reassure them that their feature engineering and model history will be kept."
        )

    # Check if user wants a full data story / comprehensive analysis
    data_story_event: dict | None = None
    if _STORY_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.storyteller import generate_data_story as _gen_story

            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _fs = ctx["feature_set"]
                _target = _fs.target_column if _fs else None
                _story = _gen_story(
                    _df,
                    dataset_id=_ds.id,
                    target_col=_target,
                    dataset_filename=_ds.filename,
                )
                data_story_event = _story
                _grade = _story["readiness_grade"]
                _score = _story["readiness_score"]
                _next = _story["recommended_next_step"]
                _sec_titles = [s["title"] for s in _story["sections"]]
                system_prompt += (
                    f"\n\n## Automated Data Story\n"
                    f"{_story['summary']}\n"
                    f"Sections analysed: {', '.join(_sec_titles)}.\n"
                    f"Recommended next step: {_next}\n"
                    "A comprehensive data story card is shown in the chat with all findings. "
                    "Summarise the key insights in plain English and guide the user to the recommended next step."
                )
        except Exception:  # noqa: BLE001
            pass  # Story is nice-to-have; never crash chat

    # Check if user wants to set a data filter ("focus on North region", "filter to Q4")
    filter_set_event: dict | None = None
    if _FILTER_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.filter_view import (
                parse_filter_request as _parse_filter,
                parse_date_filter_request as _parse_date_filter,
                apply_active_filter as _apply_filter,
                build_filter_summary as _build_filter_summary,
            )
            from models.dataset_filter import DatasetFilter as _DatasetFilter
            from db import engine as _engine

            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _full_df = pd.read_csv(_file_path)
                _all_cols = list(_full_df.columns)
                _conditions = _parse_filter(body.message, _all_cols)
                # Also try date-range parsing; merge results (date filter supplements field filters)
                _date_conditions = _parse_date_filter(body.message, _full_df)
                if _date_conditions:
                    _conditions = (_conditions or []) + _date_conditions
                if _conditions:
                    _filtered = _apply_filter(_full_df, _conditions)
                    _summary = _build_filter_summary(_conditions)
                    _orig_rows = len(_full_df)
                    _filt_rows = len(_filtered)

                    # Persist filter
                    with Session(_engine) as _fs_session:
                        _existing_af = _fs_session.exec(
                            select(_DatasetFilter).where(
                                _DatasetFilter.dataset_id == _ds.id
                            )
                        ).first()
                        if _existing_af:
                            _fs_session.delete(_existing_af)
                            _fs_session.commit()
                        _new_filter = _DatasetFilter(
                            dataset_id=_ds.id,
                            conditions=json.dumps(_conditions),
                            filter_summary=_summary,
                            original_rows=_orig_rows,
                            filtered_rows=_filt_rows,
                        )
                        _fs_session.add(_new_filter)
                        _fs_session.commit()

                    # Update the active filter for this request
                    _active_filter_conditions = _conditions
                    _reduction = round((1 - _filt_rows / max(_orig_rows, 1)) * 100, 1)
                    filter_set_event = {
                        "dataset_id": _ds.id,
                        "filter_summary": _summary,
                        "conditions": _conditions,
                        "original_rows": _orig_rows,
                        "filtered_rows": _filt_rows,
                        "row_reduction_pct": _reduction,
                    }
                    system_prompt += (
                        f"\n\n## Data Filter Applied\n"
                        f"Filter set: **{_summary}**\n"
                        f"This narrows the dataset from {_orig_rows:,} to {_filt_rows:,} rows "
                        f"({_reduction}% reduction). All subsequent analyses will use this subset. "
                        "A filter badge will appear in the Data tab. "
                        "Tell the user the filter is active and what they can now explore."
                    )
        except Exception:  # noqa: BLE001
            pass  # Filter detection is nice-to-have; never crash chat

    # Check if user wants to clear an active filter
    filter_cleared_event: dict | None = None
    if _CLEAR_FILTER_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from models.dataset_filter import DatasetFilter as _DatasetFilter
            from db import engine as _engine

            _ds = ctx["dataset"]
            with Session(_engine) as _fc_session:
                _existing_af = _fc_session.exec(
                    select(_DatasetFilter).where(_DatasetFilter.dataset_id == _ds.id)
                ).first()
                if _existing_af:
                    _fc_session.delete(_existing_af)
                    _fc_session.commit()
                    _active_filter_conditions = None
                    filter_cleared_event = {"dataset_id": _ds.id, "cleared": True}
                    system_prompt += (
                        "\n\n## Data Filter Cleared\n"
                        "The active filter has been removed. "
                        "All subsequent analyses will use the full dataset again. "
                        "Confirm to the user that the filter is cleared."
                    )
        except Exception:  # noqa: BLE001
            pass  # Filter clear is nice-to-have; never crash chat

    # Check if user wants feature selection analysis (are all columns useful?)
    feature_sel_event: dict | None = None
    if _FEATURE_SEL_PATTERNS.search(body.message) and ctx["project"]:
        try:
            from sqlmodel import select as _sel_fs

            from core.feature_engine import apply_transformations as _apply_fs
            from core.trainer import identify_weak_features as _iwf

            _proj_fs = ctx["project"]
            _ds_fs = ctx["dataset"]
            if _ds_fs:
                # Find the most recently completed model run for this project
                _done_runs = list(
                    session.exec(
                        _sel_fs(ModelRun)
                        .where(
                            ModelRun.project_id == _proj_fs.id,
                            ModelRun.status == "done",
                        )
                        .order_by(ModelRun.created_at.desc())  # type: ignore[attr-defined]
                    ).all()
                )
                _best_run: ModelRun | None = None
                for _r in _done_runs:
                    if _r.model_path and Path(_r.model_path).exists():
                        _best_run = _r
                        break

                if _best_run:
                    _fset_fs = session.exec(
                        _sel_fs(FeatureSet).where(
                            FeatureSet.dataset_id == _ds_fs.id,
                            FeatureSet.is_active == True,  # noqa: E712
                        )
                    ).first()
                    if _fset_fs and _fset_fs.target_column:
                        _fp_fs = Path(_ds_fs.file_path)
                        if _fp_fs.exists():
                            import pandas as _pd_fs

                            _df_fs = _pd_fs.read_csv(_fp_fs)
                            _tfms_fs = __import__("json").loads(
                                _fset_fs.transformations or "[]"
                            )
                            if _tfms_fs:
                                _df_fs, _ = _apply_fs(_df_fs, _tfms_fs)
                            _feat_cols_fs = [
                                c for c in _df_fs.columns if c != _fset_fs.target_column
                            ]
                            import joblib as _jl_fs

                            _model_fs = _jl_fs.load(_best_run.model_path)
                            _fs_result = _iwf(_model_fs, _feat_cols_fs)
                            feature_sel_event = {
                                "run_id": _best_run.id,
                                "algorithm": _best_run.algorithm,
                                "target_column": _fset_fs.target_column,
                                "n_features": len(_feat_cols_fs),
                                **_fs_result,
                            }
                            system_prompt += (
                                f"\n\n## Feature Selection Analysis\n"
                                f"Algorithm: {_best_run.algorithm}. "
                                f"Found {_fs_result['n_weak']} potentially weak features "
                                f"(bottom 20% by importance). "
                                f"Explanation: {_fs_result['explanation']} "
                                "Tell the user this and explain the feature selection results. "
                                "Suggest retraining without the weak features if any were found."
                            )
        except Exception:  # noqa: BLE001
            pass  # Feature selection is nice-to-have; never crash chat

    # Check if user wants to change the train/test split strategy
    split_strategy_event: dict | None = None
    if _TIME_SPLIT_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.analyzer import detect_time_columns as _dtc

            _ds_sp = ctx["dataset"]
            _file_sp = Path(_ds_sp.file_path)
            if _file_sp.exists():
                _df_sp = _load_working_df(_file_sp, _active_filter_conditions)
                _time_cols = _dtc(_df_sp)
                _wants_random = bool(
                    re.search(r"random\s+split", body.message, re.IGNORECASE)
                )
                if _wants_random:
                    split_strategy_event = {
                        "split_strategy": "random",
                        "date_col": None,
                        "explanation": (
                            "Switched to random split — rows will be shuffled and "
                            "20% held out at random for testing."
                        ),
                    }
                    system_prompt += (
                        "\n\n## Split Strategy\n"
                        "The user wants to use random (shuffled) train/test splitting. "
                        "Acknowledge that random split will be used and explain that rows "
                        "are shuffled before splitting, which is standard for non-time-series data."
                    )
                elif _time_cols:
                    split_strategy_event = {
                        "split_strategy": "chronological",
                        "date_col": _time_cols[0],
                        "explanation": (
                            f"Switched to time-based splitting on '{_time_cols[0]}' — "
                            "the model will train on older data and be tested on more recent data, "
                            "giving a more honest picture of future performance."
                        ),
                    }
                    system_prompt += (
                        f"\n\n## Split Strategy\n"
                        f"The user wants time-based splitting. The dataset has a date column "
                        f"('{_time_cols[0]}'). Explain that we'll train on the oldest 80% of "
                        f"data and test on the most recent 20% — this gives more realistic "
                        f"performance estimates for time-series forecasting than random shuffling."
                    )
                else:
                    # User asked for chronological but no date col — explain
                    system_prompt += (
                        "\n\n## Split Strategy — No Date Column\n"
                        "The user wants time-based splitting but no date column was detected. "
                        "Explain this politely and suggest using random split, or ask them to "
                        "identify which column represents dates."
                    )
        except Exception:  # noqa: BLE001
            pass

    # Check if user wants to train a model through conversation
    training_started_event: dict | None = None
    if _TRAIN_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            import queue as _queue
            import threading as _threading

            from api.models import (
                MODELS_DIR as _MODELS_DIR,
                _lock as _train_lock,
                _train_in_background,
                _training_counters,
                _training_queues,
            )
            from core.feature_engine import detect_problem_type as _detect_pt
            from core.trainer import recommend_models as _rec_models
            from models.model_run import ModelRun as _ModelRun

            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = _load_working_df(_file_path, _active_filter_conditions)
                _all_cols = list(_df.columns)
                _fs = ctx["feature_set"]

                # Resolve target: use existing, or extract from message, or ask
                _target: str | None = None
                _problem_type: str = "regression"
                _feature_set_id: str | None = None

                if _fs and _fs.target_column:
                    # Case A: already configured — use as-is
                    _target = _fs.target_column
                    _problem_type = _fs.problem_type or "regression"
                    _feature_set_id = _fs.id
                else:
                    _target_from_msg = _detect_train_target(body.message, _all_cols)
                    if _target_from_msg:
                        _target = _target_from_msg
                        _pt_result = _detect_pt(_df, _target)
                        _problem_type = _pt_result.get("problem_type", "regression")

                        if _fs:
                            # Case B: feature set exists, no target — set it
                            with Session(session.bind) as _fs_session:
                                _fs_to_update = _fs_session.get(FeatureSet, _fs.id)
                                if _fs_to_update:
                                    _fs_to_update.target_column = _target
                                    _fs_to_update.problem_type = _problem_type
                                    _fs_session.add(_fs_to_update)
                                    _fs_session.commit()
                            _feature_set_id = _fs.id
                        else:
                            # Case C: no feature set — create a minimal one
                            _feature_cols_raw = [c for c in _all_cols if c != _target]
                            _col_map = {c: [c] for c in _feature_cols_raw}
                            with Session(session.bind) as _new_fs_session:
                                _new_fs = FeatureSet(
                                    dataset_id=_ds.id,
                                    transformations=json.dumps([]),
                                    column_mapping=json.dumps(_col_map),
                                    target_column=_target,
                                    problem_type=_problem_type,
                                    is_active=True,
                                )
                                _new_fs_session.add(_new_fs)
                                _new_fs_session.commit()
                                _new_fs_session.refresh(_new_fs)
                                _feature_set_id = _new_fs.id

                if _target and _feature_set_id:
                    # Get algorithm recommendations and start training
                    _recs = _rec_models(_problem_type, _ds.row_count, _ds.column_count)
                    _algo_names = [r["algorithm"] for r in _recs[:3]]

                    _transforms_raw = (
                        json.loads(_fs.transformations or "[]") if _fs else []
                    )
                    if _transforms_raw:
                        from core.feature_engine import (
                            apply_transformations as _apply_t,
                        )

                        _df, _ = _apply_t(_df, _transforms_raw)

                    _feature_cols = [c for c in _df.columns if c != _target]
                    _model_dir = _MODELS_DIR / project_id

                    with _train_lock:
                        _training_queues[project_id] = _queue.Queue()
                        _training_counters[project_id] = len(_algo_names)

                    _run_ids: list[str] = []
                    with Session(session.bind) as _tr_session:
                        for _algo in _algo_names:
                            _run = _ModelRun(
                                project_id=project_id,
                                feature_set_id=_feature_set_id,
                                algorithm=_algo,
                                hyperparameters=json.dumps({}),
                                status="pending",
                            )
                            _tr_session.add(_run)
                            _tr_session.commit()
                            _tr_session.refresh(_run)
                            _run_ids.append(_run.id)

                            _t = _threading.Thread(
                                target=_train_in_background,
                                args=(
                                    _run.id,
                                    project_id,
                                    _df.copy(),
                                    _feature_cols,
                                    _target,
                                    _algo,
                                    _problem_type,
                                    _model_dir,
                                ),
                                daemon=True,
                            )
                            _t.start()

                    training_started_event = {
                        "project_id": project_id,
                        "target_column": _target,
                        "problem_type": _problem_type,
                        "algorithms": _algo_names,
                        "run_count": len(_run_ids),
                        "status": "started",
                    }
                    system_prompt += (
                        f"\n\n## Model Training Started\n"
                        f"Training {len(_algo_names)} model(s) to predict '{_target}' "
                        f"({_problem_type}): {', '.join(_algo_names)}.\n"
                        "Inform the user enthusiastically that training has started. "
                        "Tell them to click the Models tab to watch real-time progress. "
                        "When training finishes you'll automatically narrate the results in chat."
                    )
                elif not _target:
                    # No target found — guide the user
                    system_prompt += (
                        "\n\n## Training Request — Target Column Needed\n"
                        "The user wants to train a model but no target column has been set. "
                        "Ask them: 'What do you want to predict? For example, say "
                        "\"train a model to predict revenue\" and I'll set that up automatically.'"
                    )
        except Exception:  # noqa: BLE001
            pass  # Training initiation is nice-to-have; never crash chat

    # Check if user wants a plain-English model card / explanation
    model_card_event: dict | None = None
    if _MODEL_CARD_PATTERNS.search(body.message) and ctx["model_runs"]:
        try:
            _completed = [mr for mr in ctx["model_runs"] if mr.status == "done"]
            if _completed:
                from api.models import get_model_card as _get_model_card

                with Session(session.bind) as _mc_session:
                    _card = _get_model_card(project_id, _mc_session)
                model_card_event = _card
                system_prompt += (
                    "\n\n## Model Card\n"
                    f"Algorithm: {_card['algorithm_name']} | "
                    f"Problem type: {_card['problem_type']} | "
                    f"Target: {_card['target_col']} | "
                    f"Primary metric: {_card['metric']['display']} {_card['metric']['name']}\n"
                    f"Plain-English metric: {_card['metric']['plain_english']}\n"
                    f"Top features: {', '.join(f['feature'] for f in _card['top_features'][:3]) if _card['top_features'] else 'not available'}\n"
                    f"Key limitation: {_card['limitations'][0] if _card['limitations'] else 'none'}\n\n"
                    "The model card is being shown inline. Use this data to give a friendly, "
                    "non-technical explanation of the model. Focus on what it predicts, how "
                    "accurate it is in plain English, and the top 2-3 things driving predictions. "
                    "Keep it conversational — imagine explaining to a VP who doesn't know ML."
                )
        except Exception:  # noqa: BLE001
            pass  # Model card is nice-to-have; never crash chat

    # Check if user wants a downloadable PDF report
    report_ready_event: dict | None = None
    if _REPORT_PATTERNS.search(body.message) and ctx["model_runs"]:
        try:
            _completed_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
            _report_run = next(
                (mr for mr in _completed_runs if mr.is_selected), None
            ) or (
                max(
                    _completed_runs,
                    key=lambda r: json.loads(r.metrics or "{}").get(
                        "r2", json.loads(r.metrics or "{}").get("accuracy", 0)
                    ),
                )
                if _completed_runs
                else None
            )
            if _report_run:
                _report_metrics = json.loads(_report_run.metrics or "{}")
                _primary_metric_name = "r2" if "r2" in _report_metrics else "accuracy"
                _primary_metric_val = _report_metrics.get(_primary_metric_name)
                _report_problem_type = (
                    "regression" if "r2" in _report_metrics else "classification"
                )
                report_ready_event = {
                    "model_run_id": _report_run.id,
                    "algorithm": _report_run.algorithm,
                    "problem_type": _report_problem_type,
                    "metric_name": _primary_metric_name,
                    "metric_value": (
                        round(_primary_metric_val, 4)
                        if _primary_metric_val is not None
                        else None
                    ),
                    "download_url": f"/api/models/{_report_run.id}/report",
                }
                _metric_display = (
                    f"{_primary_metric_val:.4f}"
                    if _primary_metric_val is not None
                    else "N/A"
                )
                system_prompt += (
                    "\n\n## PDF Report Ready\n"
                    f"A PDF model report is ready for download. "
                    f"Algorithm: {_report_run.algorithm} | "
                    f"Metric: {_primary_metric_name}={_metric_display}\n"
                    "Tell the user their report is ready to download. "
                    "Briefly mention it includes model metrics, feature importance, "
                    "and confidence assessment — perfect for sharing with stakeholders."
                )
        except Exception:  # noqa: BLE001
            pass  # Report is nice-to-have; never crash chat

    # Check if user wants to deploy their model through conversation
    deployed_event: dict | None = None
    if _DEPLOY_CHAT_PATTERNS.search(body.message) and not ctx["deployment"]:
        try:
            from api.deploy import execute_deployment as _execute_deployment

            _completed_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
            _run_to_deploy = next(
                (mr for mr in _completed_runs if mr.is_selected), None
            ) or (
                max(
                    _completed_runs,
                    key=lambda r: json.loads(r.metrics or "{}").get(
                        "r2", json.loads(r.metrics or "{}").get("accuracy", 0)
                    ),
                )
                if _completed_runs
                else None
            )

            if _run_to_deploy:
                with Session(session.bind) as _dep_session:
                    _dep_result = _execute_deployment(_run_to_deploy.id, _dep_session)
                deployed_event = _dep_result
                system_prompt += (
                    "\n\n## Model Deployed!\n"
                    f"The model ({_run_to_deploy.algorithm}) has just been deployed "
                    f"and is now live. Dashboard URL: {_dep_result.get('dashboard_url')}. "
                    "Congratulate the user enthusiastically. Tell them their model is live "
                    "and they can share the dashboard link with anyone. "
                    "Mention they can also use the API endpoint to plug into their tools."
                )
            else:
                system_prompt += (
                    "\n\n## Deployment Request — No Trained Model\n"
                    "The user wants to deploy a model, but no completed training runs exist. "
                    "Tell them they need to train a model first. "
                    "Suggest: 'Say \"train a model to predict [column]\" and I'll get started.'"
                )
        except Exception:  # noqa: BLE001
            pass  # Deployment is nice-to-have; never crash chat

    # Check if user wants feature engineering suggestions
    feature_suggestions_event: dict | None = None
    if _FEATURE_SUGGEST_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.feature_engine import suggest_features as _suggest_features

            _fse_ds = ctx["dataset"]
            _fse_path = Path(_fse_ds.file_path)
            if _fse_path.exists():
                _fse_df = _load_working_df(_fse_path, _active_filter_conditions)
                _fse_col_stats = json.loads(_fse_ds.columns or "[]")
                _fse_suggestions = _suggest_features(_fse_df, _fse_col_stats)
                if _fse_suggestions:
                    feature_suggestions_event = {
                        "dataset_id": _fse_ds.id,
                        "suggestions": [
                            {
                                "id": s.id,
                                "column": s.column,
                                "transform_type": s.transform_type,
                                "title": s.title,
                                "description": s.description,
                                "preview_columns": s.preview_columns,
                            }
                            for s in _fse_suggestions
                        ],
                        "count": len(_fse_suggestions),
                    }
                    _sug_titles = ", ".join(s.title for s in _fse_suggestions[:3])
                    if len(_fse_suggestions) > 3:
                        _sug_titles += f" (+{len(_fse_suggestions) - 3} more)"
                    system_prompt += (
                        "\n\n## Feature Engineering Suggestions\n"
                        f"I've generated {len(_fse_suggestions)} feature transformation suggestions: {_sug_titles}. "
                        "Tell the user what suggestions were found and that they're shown as a card below. "
                        "Encourage them to click 'Apply All' on the card, or say 'apply features' to apply them. "
                        "Keep your reply brief — the card has all the details."
                    )
                else:
                    system_prompt += (
                        "\n\n## Feature Suggestions — None Found\n"
                        "No automatic feature transformations were found for this dataset. "
                        "The data may already be well-structured (no date columns to decompose, "
                        "no skewed numerics to log-transform, no low-cardinality categoricals to encode). "
                        "Tell the user their data appears ready to train without extra feature engineering."
                    )
        except Exception:  # noqa: BLE001
            pass  # Feature suggestions are nice-to-have; never crash chat

    # Check if user wants to apply all feature engineering suggestions
    features_applied_event: dict | None = None
    if _FEATURE_APPLY_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.feature_engine import (
                apply_transformations as _apply_transformations,
                suggest_features as _suggest_features,
            )
            from models.feature_set import FeatureSet as _FeatureSet

            _fea_ds = ctx["dataset"]
            _fea_path = Path(_fea_ds.file_path)
            if _fea_path.exists():
                _fea_df = _load_working_df(_fea_path, _active_filter_conditions)
                _fea_col_stats = json.loads(_fea_ds.columns or "[]")
                _fea_suggestions = _suggest_features(_fea_df, _fea_col_stats)
                if _fea_suggestions:
                    _fea_transforms = [
                        {"column": s.column, "transform_type": s.transform_type}
                        for s in _fea_suggestions
                    ]
                    _fea_transformed_df, _fea_mapping = _apply_transformations(
                        _fea_df, _fea_transforms
                    )
                    # Deactivate any previous active FeatureSet for this dataset
                    _fea_prev_sets = session.exec(
                        select(_FeatureSet).where(
                            _FeatureSet.dataset_id == _fea_ds.id,
                            _FeatureSet.is_active == True,  # noqa: E712
                        )
                    ).all()
                    for _fea_prev in _fea_prev_sets:
                        _fea_prev.is_active = False
                        session.add(_fea_prev)
                    _fea_new_fs = _FeatureSet(
                        dataset_id=_fea_ds.id,
                        transformations=json.dumps(_fea_transforms),
                        column_mapping=json.dumps(_fea_mapping),
                        is_active=True,
                    )
                    session.add(_fea_new_fs)
                    session.commit()
                    session.refresh(_fea_new_fs)
                    _fea_new_cols = sorted(
                        set(_fea_transformed_df.columns) - set(_fea_df.columns)
                    )
                    features_applied_event = {
                        "feature_set_id": _fea_new_fs.id,
                        "dataset_id": _fea_ds.id,
                        "new_columns": _fea_new_cols,
                        "total_columns": len(_fea_transformed_df.columns),
                        "applied_count": len(_fea_suggestions),
                    }
                    system_prompt += (
                        "\n\n## Feature Engineering Applied!\n"
                        f"I've applied {len(_fea_suggestions)} feature transformations, "
                        f"adding {len(_fea_new_cols)} new columns "
                        f"(total now: {len(_fea_transformed_df.columns)} columns). "
                        f"New columns: {', '.join(_fea_new_cols[:5])}"
                        f"{'...' if len(_fea_new_cols) > 5 else ''}. "
                        "Congratulate the user! The feature set is now active. "
                        "Suggest they can now train a model — say 'train a model to predict [column]'."
                    )
                else:
                    system_prompt += (
                        "\n\n## Feature Apply — No Suggestions Available\n"
                        "No automatic feature transformations were found for this dataset. "
                        "Tell the user their data looks ready and they can proceed directly to training."
                    )
        except Exception:  # noqa: BLE001
            pass  # Feature application is nice-to-have; never crash chat

    # Check for segment performance request ("how does my model perform by region?")
    segment_performance_event: dict | None = None
    if _SEGMENT_PERF_PATTERNS.search(body.message) and ctx["dataset"] and ctx["runs"]:
        try:
            _done_runs = [r for r in ctx["runs"] if r.status == "done"]
            _sel_run = next((r for r in _done_runs if r.is_selected), None)
            _best_run = _sel_run or (
                max(
                    _done_runs,
                    key=lambda r: (json.loads(r.metrics) if r.metrics else {}).get(
                        "r2",
                        (json.loads(r.metrics) if r.metrics else {}).get("accuracy", 0),
                    ),
                )
                if _done_runs
                else None
            )
            if (
                _best_run
                and _best_run.model_path
                and Path(_best_run.model_path).exists()
            ):
                _sp_ds = ctx["dataset"]
                _sp_file = Path(_sp_ds.file_path)
                if _sp_file.exists():
                    _sp_df_raw = pd.read_csv(_sp_file)
                    _sp_col = _detect_segment_perf_col(body.message, _sp_df_raw)
                    if _sp_col:
                        from core.validator import compute_segment_performance as _csp

                        _sp_fs = ctx.get("feature_set")
                        if _sp_fs:
                            _sp_transforms = json.loads(_sp_fs.transformations or "[]")
                            _sp_df_t = _sp_df_raw.copy()
                            if _sp_transforms:
                                from core.feature_engine import (
                                    apply_transformations as _sp_at,
                                )

                                _sp_df_t, _ = _sp_at(_sp_df_t, _sp_transforms)
                            _sp_target = _sp_fs.target_column
                            _sp_problem = _sp_fs.problem_type or "regression"
                            _sp_feat_cols = [
                                c for c in _sp_df_t.columns if c != _sp_target
                            ]
                            from core.trainer import prepare_features as _sp_pf

                            _sp_X, _sp_y, _ = _sp_pf(
                                _sp_df_t,
                                _sp_feat_cols,
                                _sp_target,
                                _sp_problem,
                            )
                            import joblib as _jl

                            _sp_model = _jl.load(_best_run.model_path)
                            _sp_y_pred = _sp_model.predict(_sp_X)
                            _sp_group_vals = _sp_df_raw[_sp_col].tolist()[: len(_sp_y)]
                            _sp_result = _csp(
                                group_values=_sp_group_vals,
                                y_true=_sp_y,
                                y_pred=_sp_y_pred,
                                problem_type=_sp_problem,
                            )
                            segment_performance_event = {
                                "group_col": _sp_col,
                                "algorithm": _best_run.algorithm,
                                "problem_type": _sp_problem,
                                **_sp_result,
                            }
                            system_prompt += (
                                f"\n\n## Model Performance by {_sp_col}\n"
                                f"{_sp_result['summary']}\n"
                                f"Metric: {_sp_result['metric_name']}. "
                                f"Best segment: '{_sp_result['best_segment']}', "
                                f"Worst segment: '{_sp_result['worst_segment']}'. "
                                "A SegmentPerformanceCard is shown in the chat. "
                                "Narrate the key insight — tell the analyst what this means for their use case "
                                "and whether they need to worry about the performance gap."
                            )
        except Exception:  # noqa: BLE001
            pass  # Segment performance is nice-to-have; never crash chat

    # Check for column profile request ("tell me about the revenue column")
    column_profile_event: dict | None = None
    if _COLUMN_PROFILE_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _cp_ds = ctx["dataset"]
            _cp_file = Path(_cp_ds.file_path)
            if _cp_file.exists():
                _cp_df = _load_working_df(_cp_file, _active_filter_conditions)
                _cp_col = _detect_profile_col(body.message, _cp_df)
                if _cp_col:
                    from core.analyzer import compute_column_profile as _ccp

                    _cp_result = _ccp(_cp_df, _cp_col)
                    if "error" not in _cp_result:
                        column_profile_event = _cp_result
                        _cp_stats = _cp_result["stats"]
                        _cp_type = _cp_result["col_type"]
                        _cp_issues = _cp_result.get("issues", [])
                        _cp_summary = _cp_result.get("summary", "")
                        system_prompt += (
                            f"\n\n## Column Profile: '{_cp_col}'\n"
                            f"Type: {_cp_type}. {_cp_summary}\n"
                            f"Stats: {json.dumps({k: v for k, v in _cp_stats.items() if k not in ('top_categories',)})}\n"
                        )
                        if _cp_issues:
                            _issue_msgs = [i["message"] for i in _cp_issues]
                            system_prompt += (
                                f"Issues detected: {'; '.join(_issue_msgs)}\n"
                            )
                        system_prompt += (
                            "A ColumnProfileCard is shown in the chat. "
                            "Narrate the key insights — help the analyst understand what this column contains, "
                            "whether there are problems, and what they should do next."
                        )
        except Exception:  # noqa: BLE001
            pass  # Column profile is nice-to-have; never crash chat

    # Check for clustering / natural segmentation request
    cluster_event: dict | None = None
    if _CLUSTER_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _cl_ds = ctx["dataset"]
            _cl_file = Path(_cl_ds.file_path)
            if _cl_file.exists():
                _cl_df = _load_working_df(_cl_file, _active_filter_conditions)
                _cl_features = _detect_cluster_features(body.message, _cl_df)
                from core.analyzer import compute_clusters as _cc

                _cl_result = _cc(_cl_df, feature_cols=_cl_features)
                if "error" not in _cl_result:
                    cluster_event = _cl_result
                    _cl_k = _cl_result["n_clusters"]
                    _cl_summary = _cl_result["summary"]
                    _cl_feat_list = ", ".join(_cl_result["features_used"][:5])
                    system_prompt += (
                        f"\n\n## K-means Clustering Result\n"
                        f"Found {_cl_k} natural groups using features: {_cl_feat_list}.\n"
                        f"Summary: {_cl_summary}\n"
                        f"Cluster descriptions:\n"
                    )
                    for _c in _cl_result["clusters"]:
                        system_prompt += f"  - {_c['description']}\n"
                    system_prompt += (
                        "A ClusteringCard is shown in the chat. "
                        "Narrate the findings — tell the analyst what each group represents, "
                        "whether the segments are actionable, and what they should do next."
                    )
        except Exception:  # noqa: BLE001
            pass  # Clustering is nice-to-have; never crash chat

    # Check for top-N / bottom-N ranking request ("top 10 customers by revenue", etc.)
    top_n_event: dict | None = None
    if _TOPN_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _tn_ds = ctx["dataset"]
            _tn_file = Path(_tn_ds.file_path)
            if _tn_file.exists():
                _tn_df = _load_working_df(_tn_file, _active_filter_conditions)
                _tn_params = _detect_topn_request(body.message, _tn_df)
                if _tn_params:
                    from core.analyzer import compute_top_n as _ctn

                    _tn_result = _ctn(
                        _tn_df,
                        _tn_params["sort_col"],
                        n=_tn_params["n"],
                        ascending=_tn_params["ascending"],
                    )
                    if "error" not in _tn_result:
                        top_n_event = _tn_result
                        _tn_dir = _tn_result["direction"]
                        _tn_col = _tn_result["sort_col"].replace("_", " ")
                        _tn_n = _tn_result["n_returned"]
                        system_prompt += (
                            f"\n\n## Top-N Ranking Result\n"
                            f"Showing the {_tn_dir} {_tn_n} records by {_tn_col}.\n"
                            f"Summary: {_tn_result['summary']}\n"
                            "A TopNCard is shown in the chat with a ranked table. "
                            "Narrate the key findings — who/what is at the top, "
                            "what patterns you notice, and what the analyst should do next."
                        )
        except Exception:  # noqa: BLE001
            pass  # Top-N ranking is nice-to-have; never crash chat

    # Check for prediction error analysis ("where was my model wrong?", "biggest errors")
    pred_error_event: dict | None = None
    if _PRED_ERROR_PATTERNS.search(body.message) and ctx["dataset"] and ctx["runs"]:
        try:
            _pe_done_runs = [r for r in ctx["runs"] if r.status == "done"]
            _pe_sel_run = next((r for r in _pe_done_runs if r.is_selected), None)
            _pe_best_run = _pe_sel_run or (
                max(
                    _pe_done_runs,
                    key=lambda r: (json.loads(r.metrics) if r.metrics else {}).get(
                        "r2",
                        (json.loads(r.metrics) if r.metrics else {}).get("accuracy", 0),
                    ),
                )
                if _pe_done_runs
                else None
            )
            if (
                _pe_best_run
                and _pe_best_run.model_path
                and Path(_pe_best_run.model_path).exists()
            ):
                _pe_fs = ctx.get("feature_set")
                _pe_ds = ctx["dataset"]
                _pe_file = Path(_pe_ds.file_path)
                if _pe_fs and _pe_file.exists():
                    import json as _pe_json

                    _pe_transforms = _pe_json.loads(_pe_fs.transformations or "[]")
                    _pe_df_raw = pd.read_csv(_pe_file)
                    _pe_df_t = _pe_df_raw.copy()
                    if _pe_transforms:
                        from core.feature_engine import (
                            apply_transformations as _pe_at,
                        )

                        _pe_df_t, _ = _pe_at(_pe_df_t, _pe_transforms)
                    _pe_target = _pe_fs.target_column
                    _pe_problem = _pe_fs.problem_type or "regression"
                    _pe_feat_cols = [c for c in _pe_df_t.columns if c != _pe_target]
                    from core.trainer import prepare_features as _pe_pf

                    _pe_X, _pe_y, _ = _pe_pf(
                        _pe_df_t, _pe_feat_cols, _pe_target, _pe_problem
                    )
                    import joblib as _pe_jl

                    _pe_model = _pe_jl.load(_pe_best_run.model_path)
                    _pe_y_pred = _pe_model.predict(_pe_X)

                    # Build display rows from the raw (pre-transform) CSV
                    _pe_display_cols = [
                        c for c in _pe_feat_cols if c in _pe_df_raw.columns
                    ]
                    _pe_feature_rows = [
                        {col: row[col] for col in _pe_display_cols if col in row}
                        for row in _pe_df_raw.head(len(_pe_y)).to_dict(orient="records")
                    ]

                    # Get class labels from pipeline if available
                    _pe_target_classes = None
                    _pe_pipeline_path = _pe_best_run.model_path.replace(
                        "_model.joblib", "_pipeline.joblib"
                    )
                    if Path(_pe_pipeline_path).exists():
                        from core.deployer import load_pipeline as _pe_lp

                        _pe_pipe = _pe_lp(_pe_pipeline_path)
                        _pe_target_classes = getattr(_pe_pipe, "target_classes", None)

                    from core.validator import compute_prediction_errors as _pe_cpe

                    _pe_n = 10  # sensible default for chat display
                    _pe_result = _pe_cpe(
                        y_true=_pe_y,
                        y_pred=_pe_y_pred,
                        problem_type=_pe_problem,
                        n=_pe_n,
                        feature_rows=_pe_feature_rows,
                        target_classes=_pe_target_classes,
                    )
                    pred_error_event = {
                        "algorithm": _pe_best_run.algorithm,
                        "target_col": _pe_target,
                        **_pe_result,
                    }
                    _pe_label = (
                        "errors" if _pe_problem == "regression" else "wrong predictions"
                    )
                    system_prompt += (
                        f"\n\n## Prediction Error Analysis\n"
                        f"Problem type: {_pe_problem}. "
                        f"Algorithm: {_pe_best_run.algorithm}. "
                        f"Summary: {_pe_result['summary']}\n"
                        f"A PredictionErrorCard is shown in chat with the top {_pe_n} "
                        f"{_pe_label}. "
                        "Narrate the key pattern: are the worst errors clustered in a "
                        "specific value range, category, or subset of the data? "
                        "Give the analyst an actionable insight, not just a description."
                    )
        except Exception:  # noqa: BLE001
            pass  # Error analysis is nice-to-have; never crash chat

    # Check for what-if / hypothetical prediction request ("what if revenue was 500?", etc.)
    whatif_chat_event: dict | None = None
    if _WHATIF_CHAT_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            _wi_deployment = ctx["deployment"]
            if (
                _wi_deployment.pipeline_path
                and Path(_wi_deployment.pipeline_path).exists()
            ):
                from core.deployer import load_pipeline as _load_pipeline
                from core.deployer import predict_single as _predict_single

                _wi_pipeline = _load_pipeline(_wi_deployment.pipeline_path)
                _wi_feature_names = _wi_pipeline.feature_names
                _wi_params = _detect_whatif_request(body.message, _wi_feature_names)
                if _wi_params:
                    _wi_feature = str(_wi_params["feature"])
                    _wi_new_value = _wi_params["new_value"]
                    # Build base from feature_means (handles unseen/zero values)
                    _wi_base: dict[str, object] = dict(_wi_pipeline.feature_means)
                    # Resolve multiplier shorthand (e.g. "double the price")
                    if isinstance(_wi_new_value, str) and _wi_new_value.startswith(
                        "__multiply__"
                    ):
                        _mult = float(_wi_new_value.split("__multiply__")[1])
                        _wi_new_value = round(
                            float(_wi_base.get(_wi_feature, 1.0)) * _mult, 4
                        )
                    # Get original prediction (from means)
                    _wi_run = next(
                        (
                            mr
                            for mr in ctx["model_runs"]
                            if mr.id == _wi_deployment.model_run_id
                        ),
                        None,
                    )
                    if (
                        _wi_run
                        and _wi_run.model_path
                        and Path(_wi_run.model_path).exists()
                    ):
                        _wi_orig = _predict_single(
                            _wi_deployment.pipeline_path,
                            _wi_run.model_path,
                            _wi_base,
                        )
                        _wi_modified_input = {**_wi_base, _wi_feature: _wi_new_value}
                        _wi_mod = _predict_single(
                            _wi_deployment.pipeline_path,
                            _wi_run.model_path,
                            _wi_modified_input,
                        )
                        _wi_orig_pred = _wi_orig["prediction"]
                        _wi_mod_pred = _wi_mod["prediction"]
                        # Compute delta
                        _wi_delta: float | None = None
                        _wi_pct: float | None = None
                        _wi_dir: str | None = None
                        try:
                            _orig_num = float(_wi_orig_pred)  # type: ignore[arg-type]
                            _mod_num = float(_wi_mod_pred)  # type: ignore[arg-type]
                            _wi_delta = round(_mod_num - _orig_num, 4)
                            _wi_pct = (
                                round((_wi_delta / (_orig_num + 1e-9)) * 100, 2)
                                if _orig_num != 0
                                else None
                            )
                            _wi_dir = (
                                "increase"
                                if _wi_delta > 0
                                else ("decrease" if _wi_delta < 0 else "no change")
                            )
                        except (TypeError, ValueError):
                            pass
                        # Build plain-English summary
                        _wi_orig_val_str = _wi_base.get(_wi_feature, "N/A")
                        if _wi_delta is not None and _wi_dir and _wi_dir != "no change":
                            _wi_summary = (
                                f"Changing {_wi_feature.replace('_', ' ')} "
                                f"from {_wi_orig_val_str} to {_wi_new_value} would "
                                f"{_wi_dir} the prediction "
                                f"from {_wi_orig_pred} to {_wi_mod_pred}"
                                + (f" ({_wi_pct:+.1f}%)" if _wi_pct is not None else "")
                                + "."
                            )
                        elif _wi_delta == 0:
                            _wi_summary = (
                                f"Changing {_wi_feature.replace('_', ' ')} "
                                f"from {_wi_orig_val_str} to {_wi_new_value} "
                                f"has no effect on the prediction ({_wi_orig_pred})."
                            )
                        else:
                            if _wi_orig_pred == _wi_mod_pred:
                                _wi_summary = (
                                    f"Changing {_wi_feature.replace('_', ' ')} "
                                    f"from {_wi_orig_val_str} to {_wi_new_value} "
                                    f"does not change the predicted class ({_wi_orig_pred})."
                                )
                            else:
                                _wi_summary = (
                                    f"Changing {_wi_feature.replace('_', ' ')} "
                                    f"from {_wi_orig_val_str} to {_wi_new_value} "
                                    f"changes the prediction from '{_wi_orig_pred}' "
                                    f"to '{_wi_mod_pred}'."
                                )
                        whatif_chat_event = {
                            "deployment_id": _wi_deployment.id,
                            "changed_feature": _wi_feature,
                            "original_feature_value": _wi_orig_val_str,
                            "new_feature_value": _wi_new_value,
                            "original_prediction": _wi_orig_pred,
                            "modified_prediction": _wi_mod_pred,
                            "delta": _wi_delta,
                            "percent_change": _wi_pct,
                            "direction": _wi_dir,
                            "summary": _wi_summary,
                            "problem_type": _wi_deployment.problem_type,
                            "target_column": _wi_deployment.target_column,
                            "original_probabilities": _wi_orig.get("probabilities"),
                            "modified_probabilities": _wi_mod.get("probabilities"),
                        }
                        system_prompt += (
                            f"\n\n## What-If Prediction Analysis\n"
                            f"The analyst asked a hypothetical: what happens if "
                            f"{_wi_feature.replace('_', ' ')} changes "
                            f"from {_wi_orig_val_str} to {_wi_new_value}?\n"
                            f"Result: {_wi_summary}\n"
                            f"A WhatIfCard is shown with the before/after comparison. "
                            f"Explain the prediction change in plain English — "
                            f"tell the analyst what this means for their decision, "
                            f"whether the change is significant, and what they might do next."
                        )
        except Exception:  # noqa: BLE001
            pass  # What-if analysis is nice-to-have; never crash chat

    # Check for inline multi-feature prediction ("predict for Region=East, Units=100, ...")
    # Distinct from what-if (single-feature): this accepts a full explicit feature set.
    # Multi-row batch prediction: "predict for: Region=East Units=100; Region=West Units=150"
    # Distinct from inline prediction (single row): uses ";" to separate multiple scenarios.
    multi_pred_event: dict | None = None
    if (
        (
            _MULTI_ROW_PRED_PATTERNS.search(body.message)
            or (";" in body.message and _INLINE_PRED_PATTERNS.search(body.message))
        )
        and ctx["deployment"]
        and not whatif_chat_event
    ):
        try:
            _mp_deployment = ctx["deployment"]
            if (
                _mp_deployment.pipeline_path
                and Path(_mp_deployment.pipeline_path).exists()
            ):
                from core.deployer import load_pipeline as _load_pipeline_mp
                from core.deployer import predict_single as _predict_single_mp

                _mp_pipeline = _load_pipeline_mp(_mp_deployment.pipeline_path)
                _mp_feature_names = _mp_pipeline.feature_names
                _mp_rows = _extract_multi_row_predictions(
                    body.message, _mp_feature_names
                )
                if _mp_rows:
                    _mp_run = next(
                        (
                            mr
                            for mr in ctx["model_runs"]
                            if mr.id == _mp_deployment.model_run_id
                        ),
                        None,
                    )
                    if (
                        _mp_run
                        and _mp_run.model_path
                        and Path(_mp_run.model_path).exists()
                    ):
                        _mp_target = _mp_deployment.target_column or "output"
                        _mp_result_rows: list[dict] = []
                        _mp_preds: list[float | str] = []
                        for _i, _row_features in enumerate(_mp_rows):
                            _mp_inputs: dict[str, object] = dict(
                                _mp_pipeline.feature_means
                            )
                            _mp_inputs.update(_row_features)
                            _r = _predict_single_mp(
                                _mp_deployment.pipeline_path,
                                _mp_run.model_path,
                                _mp_inputs,
                            )
                            _mp_pred = _r["prediction"]
                            _mp_preds.append(_mp_pred)
                            _mp_result_rows.append(
                                {
                                    "row_index": _i + 1,
                                    "provided_features": _row_features,
                                    "defaults_used_count": len(_mp_feature_names)
                                    - len(_row_features),
                                    "prediction": _mp_pred,
                                    "probabilities": _r.get("probabilities"),
                                    "confidence": _r.get("confidence"),
                                    "confidence_interval": _r.get(
                                        "confidence_interval"
                                    ),
                                }
                            )
                        # Build plain-English summary
                        _mp_n = len(_mp_result_rows)
                        _numeric_preds = [
                            p for p in _mp_preds if isinstance(p, (int, float))
                        ]
                        if _numeric_preds:
                            _mp_min = min(_numeric_preds)
                            _mp_max = max(_numeric_preds)
                            _mp_summary = (
                                f"{_mp_n} predictions for {_mp_target}: "
                                f"range {_mp_min:,.4g} – {_mp_max:,.4g}"
                            )
                        else:
                            _mp_counts: dict[str, int] = {}
                            for p in _mp_preds:
                                _mp_counts[str(p)] = _mp_counts.get(str(p), 0) + 1
                            _most_common = max(_mp_counts, key=lambda k: _mp_counts[k])
                            _mp_summary = (
                                f"{_mp_n} predictions for {_mp_target}: "
                                f"most common = {_most_common}"
                            )
                        multi_pred_event = {
                            "deployment_id": str(_mp_deployment.id),
                            "target_column": _mp_target,
                            "problem_type": _mp_deployment.problem_type,
                            "rows": _mp_result_rows,
                            "summary": _mp_summary,
                        }
                        system_prompt += (
                            f"\n\n## Multi-Row Prediction Results\n"
                            f"{_mp_summary}\n"
                            f"A MultiPredictionCard is shown in the chat with {_mp_n} rows. "
                            f"Narrate the comparison — highlight which scenario produced the "
                            f"best/worst outcome and what inputs drove the difference."
                        )
        except Exception:  # noqa: BLE001
            pass  # Multi-row prediction is nice-to-have; never crash chat

    inline_pred_event: dict | None = None
    if (
        _INLINE_PRED_PATTERNS.search(body.message)
        and ctx["deployment"]
        and not whatif_chat_event  # avoid double-predicting when what-if also fires
        and not multi_pred_event  # avoid double-predicting when multi-row already fired
    ):
        try:
            _ip_deployment = ctx["deployment"]
            if (
                _ip_deployment.pipeline_path
                and Path(_ip_deployment.pipeline_path).exists()
            ):
                from core.deployer import load_pipeline as _load_pipeline_ip
                from core.deployer import predict_single as _predict_single_ip

                _ip_pipeline = _load_pipeline_ip(_ip_deployment.pipeline_path)
                _ip_feature_names = _ip_pipeline.feature_names
                _ip_extracted = _extract_multi_feature_prediction(
                    body.message, _ip_feature_names
                )
                # Need at least one explicitly provided feature value to proceed
                if _ip_extracted:
                    # Fill missing features with training means
                    _ip_inputs: dict[str, object] = dict(_ip_pipeline.feature_means)
                    _ip_inputs.update(_ip_extracted)
                    _ip_run = next(
                        (
                            mr
                            for mr in ctx["model_runs"]
                            if mr.id == _ip_deployment.model_run_id
                        ),
                        None,
                    )
                    if (
                        _ip_run
                        and _ip_run.model_path
                        and Path(_ip_run.model_path).exists()
                    ):
                        _ip_result = _predict_single_ip(
                            _ip_deployment.pipeline_path,
                            _ip_run.model_path,
                            _ip_inputs,
                            provided_features=_ip_extracted,
                        )
                        _ip_target = _ip_deployment.target_column or "output"
                        _ip_pred = _ip_result["prediction"]
                        _ip_prob = _ip_result.get("probabilities")
                        _ip_ci = _ip_result.get("confidence_interval")
                        _ip_conf = _ip_result.get("confidence")
                        _ip_warnings = _ip_result.get("guard_rail_warnings", [])
                        # Build plain-English summary
                        _ip_used_features = list(_ip_extracted.keys())
                        _ip_defaults_count = len(_ip_feature_names) - len(
                            _ip_used_features
                        )
                        if _ip_prob:
                            # Classification: top class + probability
                            _ip_top_class = max(_ip_prob, key=lambda k: _ip_prob[k])  # type: ignore[arg-type]
                            _ip_top_pct = round(_ip_prob[_ip_top_class] * 100)  # type: ignore[index]
                            _ip_summary = (
                                f"Predicted {_ip_target}: {_ip_top_class} "
                                f"({_ip_top_pct}% probability)"
                            )
                        elif isinstance(_ip_pred, (int, float)):
                            _ip_summary = f"Predicted {_ip_target}: {_ip_pred:,.4g}"
                            if _ip_ci:
                                _ip_summary += f" (95% interval: {_ip_ci['lower']:,.4g} – {_ip_ci['upper']:,.4g})"
                        else:
                            _ip_summary = f"Predicted {_ip_target}: {_ip_pred}"
                        if _ip_defaults_count > 0:
                            _ip_summary += (
                                f". {_ip_defaults_count} feature"
                                + ("s" if _ip_defaults_count > 1 else "")
                                + " used training-data averages."
                            )
                        inline_pred_event = {
                            "deployment_id": str(_ip_deployment.id),
                            "target_column": _ip_target,
                            "prediction": _ip_pred,
                            "probabilities": _ip_prob,
                            "confidence_interval": _ip_ci,
                            "confidence": _ip_conf,
                            "provided_features": _ip_extracted,
                            "defaults_used_count": _ip_defaults_count,
                            "total_features": len(_ip_feature_names),
                            "summary": _ip_summary,
                            "problem_type": _ip_deployment.problem_type,
                            "guard_rail_warnings": _ip_warnings,
                        }
                        _ip_warn_note = ""
                        if _ip_warnings:
                            _ip_warn_note = (
                                f"\n⚠ Guard-rail warnings ({len(_ip_warnings)}): "
                                + "; ".join(w["message"] for w in _ip_warnings[:3])
                                + ". Mention these caveats when narrating."
                            )
                        system_prompt += (
                            f"\n\n## Inline Prediction Result\n"
                            f"{_ip_summary}\n"
                            f"Features provided by the analyst: "
                            f"{', '.join(f'{k}={v}' for k, v in _ip_extracted.items())}.\n"
                            f"{_ip_warn_note}\n"
                            f"An InlinePredictionCard is shown in the chat. "
                            f"Narrate the prediction in plain English — tell the analyst "
                            f"what it means in their domain context and what they might do next."
                        )
        except Exception:  # noqa: BLE001
            pass  # Inline prediction is nice-to-have; never crash chat

    # Sensitivity analysis: sweep one feature across a range → prediction curve
    sensitivity_event: dict | None = None
    if (
        _SENSITIVITY_PATTERNS.search(body.message)
        and ctx["deployment"]
        and not whatif_chat_event
        and not inline_pred_event
    ):
        try:
            _sa_deployment = ctx["deployment"]
            if (
                _sa_deployment.pipeline_path
                and Path(_sa_deployment.pipeline_path).exists()
            ):
                from core.deployer import load_pipeline as _load_pipeline_sa
                from core.deployer import run_sensitivity_analysis as _run_sa

                _sa_pipeline = _load_pipeline_sa(_sa_deployment.pipeline_path)
                _sa_feature_names = _sa_pipeline.feature_names
                _sa_means = dict(_sa_pipeline.feature_means)
                _sa_params = _detect_sensitivity_request(
                    body.message, _sa_feature_names, _sa_means
                )
                if _sa_params:
                    _sa_run = next(
                        (
                            mr
                            for mr in ctx["model_runs"]
                            if mr.id == _sa_deployment.model_run_id
                        ),
                        None,
                    )
                    if (
                        _sa_run
                        and _sa_run.model_path
                        and Path(_sa_run.model_path).exists()
                    ):
                        import numpy as np

                        _sa_sweep = list(
                            np.linspace(
                                _sa_params["min_val"],
                                _sa_params["max_val"],
                                _sa_params["n_steps"],
                            )
                        )
                        _sa_result = _run_sa(
                            _sa_deployment.pipeline_path,
                            _sa_run.model_path,
                            _sa_params["feature"],
                            _sa_sweep,
                            _sa_means,
                        )
                        sensitivity_event = _sa_result
                        system_prompt += (
                            f"\n\n## Sensitivity Analysis Result\n"
                            f"{_sa_result['summary']}\n"
                            f"A SensitivityCard is shown in the chat. "
                            f"Narrate the key finding in plain English — "
                            f"tell the analyst what this means for their business "
                            f"and whether the model is highly or weakly sensitive to this feature."
                        )
        except Exception:  # noqa: BLE001
            pass  # Sensitivity analysis is nice-to-have; never crash chat

    # Feature interaction: 2-D heatmap of two features jointly affecting prediction
    interaction_event: dict | None = None
    if (
        _INTERACTION_PATTERNS.search(body.message)
        and ctx["deployment"]
        and not sensitivity_event
        and not whatif_chat_event
        and not inline_pred_event
    ):
        try:
            _ia_deployment = ctx["deployment"]
            if (
                _ia_deployment.pipeline_path
                and Path(_ia_deployment.pipeline_path).exists()
            ):
                from core.deployer import load_pipeline as _load_pipeline_ia
                from core.deployer import run_feature_interaction as _run_ia

                _ia_pipeline = _load_pipeline_ia(_ia_deployment.pipeline_path)
                _ia_feature_names = _ia_pipeline.feature_names
                _ia_means = dict(_ia_pipeline.feature_means)
                _ia_req = _detect_interaction_request(body.message, _ia_feature_names)
                if _ia_req:
                    _ia_run = next(
                        (
                            mr
                            for mr in ctx["model_runs"]
                            if mr.id == _ia_deployment.model_run_id
                        ),
                        None,
                    )
                    if (
                        _ia_run
                        and _ia_run.model_path
                        and Path(_ia_run.model_path).exists()
                    ):
                        _ia_result = _run_ia(
                            _ia_deployment.pipeline_path,
                            _ia_run.model_path,
                            _ia_req["feature1"],
                            _ia_req["feature2"],
                            _ia_means,
                        )
                        interaction_event = _ia_result
                        system_prompt += (
                            f"\n\n## Feature Interaction Result\n"
                            f"{_ia_result['summary']}\n"
                            f"An InteractionCard (2-D heatmap) is shown in the chat. "
                            f"Narrate the key finding in plain English — tell the analyst "
                            f"which combination of {_ia_req['feature1']} and "
                            f"{_ia_req['feature2']} produces the best (or worst) outcome, "
                            f"and whether the two features interact or are mostly independent."
                        )
        except Exception:  # noqa: BLE001
            pass  # Interaction analysis is nice-to-have; never crash chat

    # Dataset ranking — "which customers are most likely to churn?" / "show me top 20"
    ranked_pred_event: dict | None = None
    if (
        _RANKED_PRED_PATTERNS.search(body.message)
        and ctx["deployment"]
        and ctx["dataset"]
        and not interaction_event
        and not sensitivity_event
        and not whatif_chat_event
        and not inline_pred_event
    ):
        try:
            _rp_deployment = ctx["deployment"]
            _rp_ds = ctx["dataset"]
            _rp_file = Path(_rp_ds.file_path)
            if (
                _rp_deployment.pipeline_path
                and Path(_rp_deployment.pipeline_path).exists()
                and _rp_file.exists()
            ):
                _rp_run = next(
                    (
                        mr
                        for mr in ctx["model_runs"]
                        if mr.id == _rp_deployment.model_run_id
                    ),
                    None,
                )
                if _rp_run and _rp_run.model_path and Path(_rp_run.model_path).exists():
                    from core.deployer import run_dataset_ranking as _run_ranking

                    _rp_df = _load_working_df(_rp_file, _active_filter_conditions)
                    _rp_req = _detect_ranked_pred_request(body.message)
                    _rp_result = _run_ranking(
                        _rp_deployment.pipeline_path,
                        _rp_run.model_path,
                        _rp_df,
                        n=_rp_req["n"],
                        direction=_rp_req["direction"],
                    )
                    ranked_pred_event = _rp_result
                    system_prompt += (
                        f"\n\n## Dataset Ranking Result\n"
                        f"{_rp_result['summary']}\n"
                        f"A RankedPredictionsCard is shown in the chat with the top "
                        f"{_rp_result['n']} rows ranked by predicted "
                        f"{_rp_result['target_column']} ({_rp_result['direction']} first). "
                        f"Narrate this in plain English — tell the analyst which rows to "
                        f"focus on and what the predictions mean for their business."
                    )
        except Exception:  # noqa: BLE001
            pass  # Ranking is nice-to-have; never crash chat

    # Prediction cohort analysis — "who are the top predictions?" / "profile at-risk customers"
    cohort_event: dict | None = None
    if (
        _COHORT_PATTERNS.search(body.message)
        and ctx["deployment"]
        and ctx["dataset"]
        and not ranked_pred_event
    ):
        try:
            _co_deployment = ctx["deployment"]
            _co_ds = ctx["dataset"]
            _co_file = Path(_co_ds.file_path)
            if (
                _co_deployment.pipeline_path
                and Path(_co_deployment.pipeline_path).exists()
                and _co_file.exists()
            ):
                _co_run = next(
                    (
                        mr
                        for mr in ctx["model_runs"]
                        if mr.id == _co_deployment.model_run_id
                    ),
                    None,
                )
                if _co_run and _co_run.model_path and Path(_co_run.model_path).exists():
                    from core.deployer import compute_prediction_cohort as _cpc

                    _co_df = _load_working_df(_co_file, _active_filter_conditions)
                    _co_req = _detect_ranked_pred_request(body.message)
                    _co_result = _cpc(
                        _co_deployment.pipeline_path,
                        _co_run.model_path,
                        _co_df,
                        n=_co_req["n"],
                        direction=_co_req["direction"],
                    )
                    cohort_event = _co_result
                    system_prompt += (
                        f"\n\n## Prediction Cohort Profile\n"
                        f"{_co_result['characterization']}\n"
                        f"A PredictionCohortCard is shown in the chat profiling the top "
                        f"{_co_result['n']} {_co_result['direction']}-scoring "
                        f"{_co_result['target_column']} predictions. "
                        f"Narrate the key distinguishing traits of this group — what makes "
                        f"them different from the overall dataset, and what the analyst should "
                        f"do with this insight."
                    )
        except Exception:  # noqa: BLE001
            pass  # Cohort profiling is nice-to-have; never crash chat

    # Partial Dependence Plot — "marginal effect of price", "PDP for units"
    # Distinct from sensitivity analysis (which holds others at training means).
    # PDP averages over the ACTUAL training distribution → more accurate marginal effect.
    pdp_event: dict | None = None
    if (
        _PDP_PATTERNS.search(body.message)
        and ctx.get("model_runs")
        and not sensitivity_event
        and not interaction_event
    ):
        try:
            import json as _json

            import joblib as _jl
            import numpy as _np

            from core.explainer import compute_partial_dependence as _cpd
            from core.feature_engine import apply_transformations as _at
            from core.trainer import prepare_features as _pf

            # Pick best/selected completed run
            _pdp_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
            _pdp_run = next(
                (mr for mr in _pdp_runs if mr.is_selected),
                _pdp_runs[0] if _pdp_runs else None,
            )
            if (
                _pdp_run
                and _pdp_run.model_path
                and Path(_pdp_run.model_path).exists()
                and ctx["dataset"]
                and ctx.get("feature_set")
            ):
                _pdp_fs = ctx["feature_set"]
                _pdp_ds = ctx["dataset"]
                _pdp_file = Path(_pdp_ds.file_path)
                if _pdp_file.exists():
                    import pandas as _pd_pdp

                    _pdp_df = _pd_pdp.read_csv(_pdp_file)
                    _pdp_transforms = _json.loads(_pdp_fs.transformations or "[]")
                    if _pdp_transforms:
                        _pdp_df, _ = _at(_pdp_df, _pdp_transforms)
                    _pdp_target = _pdp_fs.target_column
                    _pdp_feat_cols = [c for c in _pdp_df.columns if c != _pdp_target]
                    _pdp_prob_type = _pdp_fs.problem_type or "regression"

                    _pdp_X, _pdp_y, _pdp_feature_names = _pf(
                        _pdp_df,
                        _pdp_feat_cols,
                        _pdp_target,
                        _pdp_prob_type,
                    )

                    _pdp_feat = _detect_pdp_feature(body.message, _pdp_feature_names)
                    if _pdp_feat is None and _pdp_feature_names:
                        _pdp_feat = _pdp_feature_names[0]

                    if _pdp_feat and _pdp_feat in _pdp_feature_names:
                        _pdp_idx = _pdp_feature_names.index(_pdp_feat)
                        _pdp_col_vals = _pdp_X[:, _pdp_idx]
                        _pdp_p5 = float(_np.percentile(_pdp_col_vals, 5))
                        _pdp_p95 = float(_np.percentile(_pdp_col_vals, 95))
                        _pdp_steps = 20
                        if _pdp_p5 == _pdp_p95:
                            _pdp_grid = _np.array([_pdp_p5])
                        else:
                            _pdp_grid = _np.linspace(_pdp_p5, _pdp_p95, _pdp_steps)

                        # Resolve class names
                        _pdp_class_names = None
                        _pdp_pipeline_path = _pdp_run.model_path.replace(
                            "_model.joblib", "_pipeline.joblib"
                        )
                        if Path(_pdp_pipeline_path).exists():
                            try:
                                _pdp_pipe = _jl.load(_pdp_pipeline_path)
                                _pdp_class_names = getattr(
                                    _pdp_pipe, "target_classes", None
                                )
                            except Exception:  # noqa: BLE001
                                pass

                        _pdp_model = _jl.load(_pdp_run.model_path)
                        _pdp_result = _cpd(
                            model=_pdp_model,
                            X_train=_pdp_X,
                            feature_idx=_pdp_idx,
                            grid_values=_pdp_grid,
                            problem_type=_pdp_prob_type,
                            class_names=_pdp_class_names,
                        )
                        pdp_event = {
                            **_pdp_result,
                            "feature": _pdp_feat,
                            "target_col": _pdp_target,
                            "algorithm": _pdp_run.algorithm,
                        }
                        system_prompt += (
                            f"\n\n## Partial Dependence Analysis\n"
                            f"Feature swept: {_pdp_feat} (from {_pdp_grid[0]:.4g} to {_pdp_grid[-1]:.4g})\n"
                            f"{_pdp_result['summary']}\n"
                            f"A PartialDependenceCard is shown in the chat with a line chart "
                            f"of the average predicted {_pdp_target} across {_pdp_result['n_training_rows']} "
                            f"training records as {_pdp_feat} varies. "
                            f"Unlike sensitivity analysis (which fixes other features at their means), "
                            f"this PDP averages over the actual training data distribution — giving a "
                            f"more accurate marginal effect. Narrate the key finding in plain English: "
                            f"does increasing {_pdp_feat} consistently raise or lower the prediction, "
                            f"is there a threshold or nonlinear effect, and what should the analyst do with this?"
                        )
        except Exception:  # noqa: BLE001
            pass  # PDP is nice-to-have; never crash chat

    # Calibration check — reliability of confidence scores for classifiers
    calibration_check_event: dict | None = None
    if _CALIBRATION_CHECK_PATTERNS.search(body.message) and ctx.get("model_runs"):
        try:
            import json as _json_cal

            _cal_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
            _cal_run = next(
                (mr for mr in _cal_runs if mr.is_selected),
                _cal_runs[0] if _cal_runs else None,
            )
            if _cal_run:
                _cal_metrics = _json_cal.loads(_cal_run.metrics or "{}")
                if _cal_metrics.get("is_calibrated"):
                    _cal_curve = _cal_metrics.get("calibration_curve", [])
                    _cal_brier = _cal_metrics.get("brier_score")
                    _cal_note = _cal_metrics.get("calibration_note", "")
                    # Compute plain-English calibration quality summary
                    _cal_quality = "unknown"
                    _cal_summary = ""
                    if _cal_brier is not None:
                        if _cal_brier < 0.1:
                            _cal_quality = "excellent"
                            _cal_summary = (
                                f"Brier score {_cal_brier:.3f} — excellent calibration. "
                                "When the model says '80% confident', it's right roughly 80% of the time."
                            )
                        elif _cal_brier < 0.2:
                            _cal_quality = "good"
                            _cal_summary = (
                                f"Brier score {_cal_brier:.3f} — good calibration. "
                                "Confidence scores are generally reliable but may be slightly over- or under-confident."
                            )
                        else:
                            _cal_quality = "poor"
                            _cal_summary = (
                                f"Brier score {_cal_brier:.3f} — calibration needs attention. "
                                "The model's stated confidence may not match actual accuracy at each level."
                            )
                    calibration_check_event = {
                        "run_id": _cal_run.id,
                        "algorithm": _cal_run.algorithm,
                        "is_calibrated": True,
                        "brier_score": _cal_brier,
                        "calibration_quality": _cal_quality,
                        "calibration_curve": _cal_curve,
                        "calibration_note": _cal_note,
                        "summary": _cal_summary,
                    }
                    system_prompt += (
                        f"\n\n## Model Calibration Analysis\n"
                        f"Algorithm: {_cal_run.algorithm}\n"
                        f"{_cal_summary}\n"
                        f"{_cal_note}\n"
                        f"A CalibrationCheckCard is shown with the reliability diagram. "
                        f"Explain what calibration means in plain English (a calibrated model's "
                        f"stated 70% confidence matches real 70% accuracy), describe the Brier score "
                        f"result, and advise whether the analyst can trust the confidence scores shown "
                        f"on the prediction dashboard."
                    )
                else:
                    # Classifier exists but calibration not available — explain why
                    _cal_problem = _cal_metrics.get("problem_type", "")
                    if _cal_problem == "regression":
                        system_prompt += (
                            "\n\n## Calibration Not Available\n"
                            "Calibration curves apply to classifiers (models that output probabilities). "
                            "This is a regression model — it predicts numeric values, not probabilities. "
                            "Calibration does not apply."
                        )
                    else:
                        system_prompt += (
                            "\n\n## Calibration Not Available\n"
                            "Calibration data was not computed for this model run. "
                            "This can happen when threshold tuning, SMOTE, or too few training rows were used. "
                            "Explain this to the analyst and suggest retraining with more data if needed."
                        )
        except Exception:  # noqa: BLE001
            pass  # Calibration check is nice-to-have; never crash chat

    # Guided onboarding wizard — responds to "guide me", "first steps", etc.
    onboarding_event: dict | None = None
    if _ONBOARDING_PATTERNS.search(body.message):
        try:
            from core.onboarding import compute_onboarding_state as _cos

            _ob_has_dataset = ctx["dataset"] is not None
            _ob_messages: list = []
            if ctx.get("conversation"):
                import json as _json_ob

                _ob_messages = _json_ob.loads(ctx["conversation"].messages or "[]")
            _ob_msg_count = len(_ob_messages)
            _ob_fs = ctx.get("feature_set")
            _ob_has_target = bool(_ob_fs and _ob_fs.target_column)
            _ob_runs = ctx.get("model_runs") or []
            _ob_done_runs = [r for r in _ob_runs if r.status == "done"]
            _ob_has_run = len(_ob_done_runs) > 0
            _ob_has_cv = any(
                (
                    json.loads(r.metrics or "{}").get("cv_r2_mean") is not None
                    or json.loads(r.metrics or "{}").get("cv_accuracy_mean") is not None
                )
                for r in _ob_done_runs
            )
            _ob_has_deploy = ctx["deployment"] is not None
            onboarding_event = _cos(
                has_dataset=_ob_has_dataset,
                message_count=_ob_msg_count,
                has_target=_ob_has_target,
                has_model_run=_ob_has_run,
                has_cross_val=_ob_has_cv,
                has_deployment=_ob_has_deploy,
            )
            system_prompt += (
                f"\n\n## Analyst Onboarding State\n"
                f"Step {onboarding_event['step_index'] + 1} of "
                f"{onboarding_event['total_steps']}: "
                f"{onboarding_event['summary']}\n"
                f"An OnboardingGuideCard is shown with the step list. "
                f"Acknowledge their progress warmly, name the current step, "
                f"and give one concrete tip for completing it — no bullet lists, "
                f"one natural paragraph."
            )
        except Exception:  # noqa: BLE001
            pass  # Onboarding card is nice-to-have; never crash chat

    # Learning curve analysis — "would more data help?", "learning curve"
    learning_curve_event: dict | None = None
    if _LEARNING_CURVE_PATTERNS.search(body.message) and ctx.get("model_runs"):
        try:
            _lc_runs = [r for r in (ctx.get("model_runs") or []) if r.status == "done"]
            if _lc_runs:
                # Prefer selected; else best by primary metric
                _lc_run = next((r for r in _lc_runs if r.is_selected), None)
                if not _lc_run:
                    from api.models import REGRESSION_ALGORITHMS as _REG_ALGOS
                    from api.models import CLASSIFICATION_ALGORITHMS as _CLS_ALGOS

                    _lc_reg = [r for r in _lc_runs if r.algorithm in _REG_ALGOS]
                    _lc_cls = [r for r in _lc_runs if r.algorithm in _CLS_ALGOS]
                    if _lc_reg:
                        _lc_run = max(
                            _lc_reg,
                            key=lambda r: json.loads(r.metrics or "{}").get("r2", 0),
                        )
                    elif _lc_cls:
                        _lc_run = max(
                            _lc_cls,
                            key=lambda r: json.loads(r.metrics or "{}").get(
                                "accuracy", 0
                            ),
                        )
                    else:
                        _lc_run = _lc_runs[-1]

                _lc_ds = ctx["dataset"]
                if _lc_ds and _lc_ds.file_path and Path(_lc_ds.file_path).exists():
                    _lc_df = pd.read_csv(_lc_ds.file_path)
                    _lc_fs = ctx.get("feature_set")
                    if _lc_fs:
                        _lc_transforms = json.loads(_lc_fs.transformations or "[]")
                        if _lc_transforms:
                            from core.feature_engine import (
                                apply_transformations as _lc_at,
                            )

                            _lc_df, _ = _lc_at(_lc_df, _lc_transforms)
                        _lc_target = _lc_fs.target_column
                    else:
                        _lc_target = _lc_df.columns[-1]

                    if _lc_target and _lc_target in _lc_df.columns:
                        from api.models import (
                            CLASSIFICATION_ALGORITHMS as _LC_CLS_ALGOS,
                        )

                        _lc_problem = (
                            "classification"
                            if _lc_run.algorithm in _LC_CLS_ALGOS
                            else "regression"
                        )
                        _lc_feat_cols = [c for c in _lc_df.columns if c != _lc_target]
                        from core.trainer import (
                            compute_learning_curve as _clc,
                            prepare_features as _lc_pf,
                        )

                        _lc_X, _lc_y, _ = _lc_pf(
                            _lc_df, _lc_feat_cols, _lc_target, _lc_problem
                        )
                        _lc_result = _clc(_lc_X, _lc_y, _lc_run.algorithm, _lc_problem)
                        learning_curve_event = _lc_result
                        system_prompt += (
                            f"\n\n## Learning Curve Analysis\n"
                            f"{_lc_result['summary']}\n"
                            f"Recommendation: {_lc_result['recommendation']}\n"
                            f"A LearningCurveCard is shown in the chat with a chart. "
                            f"Narrate the key insight in plain English — "
                            f"should the analyst collect more data or focus on better features? "
                            f"Be specific and actionable."
                        )
        except Exception:  # noqa: BLE001
            pass  # Learning curve is nice-to-have; never crash chat

    # Check for "show me the data" / record table viewer
    records_event: dict | None = None
    if _RECORDS_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _rec_ds = ctx["dataset"]
            _rec_file = Path(_rec_ds.file_path)
            if _rec_file.exists():
                _rec_df = _load_working_df(_rec_file, _active_filter_conditions)
                _rec_params = _detect_records_request(body.message, _rec_df)
                from core.analyzer import sample_records as _sample_records

                _rec_result = _sample_records(
                    _rec_df,
                    n=_rec_params["n"],
                    conditions=_rec_params["conditions"],
                )
                records_event = _rec_result
                system_prompt += (
                    f"\n\n## Data Sample\n"
                    f"{_rec_result['summary']}\n"
                    f"A RecordTableCard is shown in the chat with "
                    f"{_rec_result['shown_rows']} rows "
                    f"({'filtered by: ' + _rec_result['condition_summary'] if _rec_result['filtered'] else 'from the beginning of the dataset'}).\n"
                    "Briefly narrate what the analyst is looking at — mention a few "
                    "notable values if any stand out, and suggest a next analysis step."
                )
        except Exception:  # noqa: BLE001
            pass  # Record preview is nice-to-have; never crash chat

    # Check for time-period comparison request ("compare 2023 vs 2024", "Q1 vs Q2", etc.)
    time_window_event: dict | None = None
    if _TIMEWINDOW_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _tw_ds = ctx["dataset"]
            _tw_file = Path(_tw_ds.file_path)
            if _tw_file.exists():
                _tw_df = _load_working_df(_tw_file, _active_filter_conditions)
                _tw_params = _detect_timewindow_request(body.message, _tw_df)
                if _tw_params:
                    from core.analyzer import compare_time_windows as _ctw

                    _tw_result = _ctw(
                        _tw_df,
                        _tw_params["date_col"],
                        _tw_params["period1_name"],
                        _tw_params["period1_start"],
                        _tw_params["period1_end"],
                        _tw_params["period2_name"],
                        _tw_params["period2_start"],
                        _tw_params["period2_end"],
                    )
                    if "error" not in _tw_result:
                        time_window_event = _tw_result
                        system_prompt += (
                            f"\n\n## Time-Period Comparison\n"
                            f"Comparing {_tw_result['period1']['name']} "
                            f"({_tw_result['period1']['row_count']} rows) vs "
                            f"{_tw_result['period2']['name']} "
                            f"({_tw_result['period2']['row_count']} rows).\n"
                            f"Summary: {_tw_result['summary']}\n"
                        )
                        if _tw_result["notable_changes"]:
                            system_prompt += (
                                f"Notable changes (>20%): "
                                f"{', '.join(_tw_result['notable_changes'])}.\n"
                            )
                        system_prompt += (
                            "A TimeWindowCard is shown in the chat. "
                            "Narrate the key findings — highlight the biggest changes, "
                            "whether the trend is positive or negative, and what the analyst should do next."
                        )
        except Exception:  # noqa: BLE001
            pass  # Time-window comparison is nice-to-have; never crash chat

    # Check if user wants to save an analysis template
    template_saved_event: dict | None = None
    if _SAVE_TEMPLATE_PATTERNS.search(body.message) and project:
        try:
            from models.analysis_template import AnalysisTemplate as _AT_Model

            _tpl_name = _extract_template_name(body.message) or "My Analysis"
            # Load recent user queries from conversation history (last 8, skip the save msg)
            _conv_save = session.exec(
                select(Conversation).where(Conversation.project_id == body.project_id)
            ).first()
            _user_queries: list[str] = []
            if _conv_save:
                _all_msgs = json.loads(_conv_save.messages)
                _user_queries = [
                    m["content"]
                    for m in _all_msgs
                    if m.get("role") == "user"
                    and not _SAVE_TEMPLATE_PATTERNS.search(m.get("content", ""))
                ][-8:]
            if _user_queries:
                _new_tpl = _AT_Model(
                    project_id=body.project_id,
                    name=_tpl_name,
                    queries=json.dumps(_user_queries),
                )
                with Session(session.bind) as _tpl_sess:
                    _tpl_sess.add(_new_tpl)
                    _tpl_sess.commit()
                    _tpl_sess.refresh(_new_tpl)
                template_saved_event = {
                    "id": _new_tpl.id,
                    "name": _tpl_name,
                    "queries": _user_queries,
                    "query_count": len(_user_queries),
                }
                system_prompt += (
                    f"\n\n## Analysis Template Saved\n"
                    f"Saved {len(_user_queries)} queries as template '{_tpl_name}'. "
                    "Confirm this to the user. Tell them they can replay this template "
                    "anytime by saying 'replay my [template name] template'."
                )
        except Exception:  # noqa: BLE001
            pass  # Template saving is nice-to-have; never crash chat

    # Check if user wants to list their saved templates
    template_list_event: dict | None = None
    if _LIST_TEMPLATES_PATTERNS.search(body.message) and project:
        try:
            from models.analysis_template import AnalysisTemplate as _AT_List

            _templates = session.exec(
                select(_AT_List)
                .where(_AT_List.project_id == body.project_id)
                .order_by(_AT_List.created_at.desc())
            ).all()
            _tpl_list = [
                {
                    "id": t.id,
                    "name": t.name,
                    "queries": json.loads(t.queries) if t.queries else [],
                    "query_count": len(json.loads(t.queries)) if t.queries else 0,
                    "created_at": t.created_at.isoformat(),
                }
                for t in _templates
            ]
            template_list_event = {
                "templates": _tpl_list,
                "count": len(_tpl_list),
            }
            if _tpl_list:
                _names = ", ".join(f"'{t['name']}'" for t in _tpl_list[:3])
                system_prompt += (
                    f"\n\n## Saved Analysis Templates\n"
                    f"{len(_tpl_list)} template(s) saved: {_names}. "
                    "Show the list to the user and tell them they can replay any template "
                    "by saying 'replay my [name] template'."
                )
            else:
                system_prompt += (
                    "\n\n## Saved Analysis Templates\n"
                    "No templates saved yet. "
                    "Tell the user they can save their analysis queries as a template by "
                    "saying 'save this analysis as a template called [name]'."
                )
        except Exception:  # noqa: BLE001
            pass  # Template listing is nice-to-have; never crash chat

    # Check if user wants to replay a saved template
    template_replay_event: dict | None = None
    if _REPLAY_TEMPLATE_PATTERNS.search(body.message) and project:
        try:
            from models.analysis_template import AnalysisTemplate as _AT_Replay

            _replay_name = _extract_template_name(body.message)
            _all_tpls = session.exec(
                select(_AT_Replay)
                .where(_AT_Replay.project_id == body.project_id)
                .order_by(_AT_Replay.created_at.desc())
            ).all()
            _matched_tpl: "_AT_Replay | None" = None
            if _replay_name:
                _replay_lower = _replay_name.lower()
                for _t in _all_tpls:
                    if _replay_lower in _t.name.lower():
                        _matched_tpl = _t
                        break
            if _matched_tpl is None and _all_tpls:
                _matched_tpl = _all_tpls[0]  # Fall back to most recent
            if _matched_tpl:
                _replay_queries = (
                    json.loads(_matched_tpl.queries) if _matched_tpl.queries else []
                )
                template_replay_event = {
                    "id": _matched_tpl.id,
                    "name": _matched_tpl.name,
                    "queries": _replay_queries,
                    "query_count": len(_replay_queries),
                }
                system_prompt += (
                    f"\n\n## Template Replay\n"
                    f"Replaying template '{_matched_tpl.name}' with {len(_replay_queries)} queries. "
                    "Tell the user their template queries are shown as clickable buttons — "
                    "they can click each query to re-run it on their current data."
                )
        except Exception:  # noqa: BLE001
            pass  # Template replay is nice-to-have; never crash chat

    # Check if user wants to save a prediction preset
    preset_saved_event: dict | None = None
    if _PRESET_SAVE_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            _preset_def = _extract_preset_definition(body.message)
            if _preset_def:
                from models.deployment_preset import DeploymentPreset as _PresetModel

                _new_preset = _PresetModel(
                    deployment_id=ctx["deployment"].id,
                    name=_preset_def["name"],
                    feature_values=json.dumps(_preset_def["feature_values"]),
                )
                with Session(session.bind) as _preset_sess:
                    _preset_sess.add(_new_preset)
                    _preset_sess.commit()
                    _preset_sess.refresh(_new_preset)
                preset_saved_event = {
                    "id": _new_preset.id,
                    "deployment_id": ctx["deployment"].id,
                    "name": _preset_def["name"],
                    "feature_values": _preset_def["feature_values"],
                    "feature_count": len(_preset_def["feature_values"]),
                }
                system_prompt += (
                    f"\n\n## Prediction Preset Saved\n"
                    f"Saved preset '{_preset_def['name']}' with "
                    f"{len(_preset_def['feature_values'])} feature values: "
                    + ", ".join(
                        f"{k}={v}" for k, v in _preset_def["feature_values"].items()
                    )
                    + ". "
                    "Tell the user this preset now appears as a quick-fill button on the "
                    "shared prediction dashboard. VPs and colleagues can click it to "
                    "instantly fill the form with these values."
                )
        except Exception:  # noqa: BLE001
            pass  # Preset saving is nice-to-have; never crash chat

    # Check if user wants to list saved presets
    preset_list_event: dict | None = None
    if _PRESET_LIST_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            from models.deployment_preset import DeploymentPreset as _PresetList

            _saved_presets = session.exec(
                select(_PresetList)
                .where(_PresetList.deployment_id == ctx["deployment"].id)
                .order_by(_PresetList.created_at)
            ).all()
            _preset_items = [
                {
                    "id": p.id,
                    "name": p.name,
                    "feature_values": json.loads(p.feature_values),
                    "feature_count": len(json.loads(p.feature_values)),
                }
                for p in _saved_presets
            ]
            preset_list_event = {
                "presets": _preset_items,
                "count": len(_preset_items),
                "deployment_id": ctx["deployment"].id,
            }
            if _preset_items:
                _pnames = ", ".join(f"'{p['name']}'" for p in _preset_items[:4])
                system_prompt += (
                    f"\n\n## Saved Prediction Presets\n"
                    f"{len(_preset_items)} preset(s) saved: {_pnames}. "
                    "Each preset appears as a quick-fill button on the shared prediction dashboard. "
                    "Show the list to the user."
                )
            else:
                system_prompt += (
                    "\n\n## Saved Prediction Presets\n"
                    "No presets saved yet for this deployment. "
                    "Tell the user they can save a preset by saying: "
                    "'add a preset called [Name] with [feature=value, ...]'"
                )
        except Exception:  # noqa: BLE001
            pass  # Preset listing is nice-to-have; never crash chat

    # SDK download card: generate download links for Python/JS SDK
    sdk_event: dict | None = None
    if _SDK_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            _dep = ctx["deployment"]
            _target = _dep.target_column or "target"
            _algo = _dep.algorithm or "model"
            _problem = _dep.problem_type or "regression"
            # Derive class name the same way the endpoint does
            _parts = (_target + "_predictor").replace("-", "_").split("_")
            _class_name = "".join(p.capitalize() for p in _parts if p)
            _base = f"/api/deploy/{_dep.id}/sdk"
            sdk_event = {
                "deployment_id": _dep.id,
                "target_column": _target,
                "algorithm": _algo,
                "problem_type": _problem,
                "python_url": f"{_base}?language=python",
                "javascript_url": f"{_base}?language=javascript",
                "class_name": _class_name,
            }
            system_prompt += (
                f"\n\n## SDK Download Ready\n"
                f"A downloadable SDK has been generated for the {_target} model "
                f"({_algo.replace('_', ' ').title()}, {_problem}). "
                f"The SDK class is called `{_class_name}`. "
                "It wraps the prediction API in a typed Python or JavaScript class. "
                "Tell the user they can download the Python SDK or JavaScript SDK "
                "using the download buttons in the card. "
                "The SDK lets their developer import the class and call predict() "
                "without writing HTTP code manually."
            )
        except Exception:  # noqa: BLE001
            pass  # SDK generation is nice-to-have; never crash chat

    # Cross-project portfolio overview
    portfolio_event: dict | None = None
    if _PORTFOLIO_PATTERNS.search(body.message):
        try:
            from core.analyzer import compute_portfolio_summary as _cps

            _all_projects = list(session.exec(select(Project)).all())
            _project_summaries: list[dict] = []
            for _proj in _all_projects:
                _ds = session.exec(
                    select(Dataset).where(Dataset.project_id == _proj.id)
                ).first()
                _runs = list(
                    session.exec(
                        select(ModelRun).where(
                            ModelRun.project_id == _proj.id,
                            ModelRun.status == "done",
                        )
                    ).all()
                )
                _dep = session.exec(
                    select(Deployment).where(
                        Deployment.project_id == _proj.id,
                        Deployment.is_active == True,  # noqa: E712
                    )
                ).first()
                _pred_count = 0
                if _dep:
                    _pred_count = len(
                        list(
                            session.exec(
                                select(PredictionLog).where(
                                    PredictionLog.deployment_id == _dep.id
                                )
                            ).all()
                        )
                    )
                _best_run = None
                _best_val: float | None = None
                _best_metric: str | None = None
                for _run in _runs:
                    _m = json.loads(_run.metrics or "{}")
                    _v = _m.get("r2") or _m.get("accuracy")
                    _mn = (
                        "r2"
                        if "r2" in _m
                        else ("accuracy" if "accuracy" in _m else None)
                    )
                    if _v is not None and (_best_val is None or _v > _best_val):
                        _best_val = _v
                        _best_metric = _mn
                        _best_run = _run
                _project_summaries.append(
                    {
                        "project_id": _proj.id,
                        "name": _proj.name,
                        "dataset_filename": _ds.filename if _ds else None,
                        "row_count": _ds.row_count if _ds else None,
                        "model_count": len(_runs),
                        "best_algorithm": _best_run.algorithm if _best_run else None,
                        "best_metric_name": _best_metric,
                        "best_metric_value": _best_val,
                        "best_problem_type": (
                            _best_run.problem_type if _best_run else None
                        ),
                        "best_target_column": (
                            _best_run.target_column if _best_run else None
                        ),
                        "has_deployment": _dep is not None,
                        "prediction_count": _pred_count,
                        "last_activity_at": (
                            _proj.updated_at.isoformat() if _proj.updated_at else None
                        ),
                    }
                )
            portfolio_event = _cps(_project_summaries)
            system_prompt += (
                f"\n\n## Portfolio Overview\n"
                f"{portfolio_event['summary']}\n"
                f"Total projects: {portfolio_event['total_projects']}. "
                f"Active deployments: {portfolio_event['active_deployments']}. "
                f"Total predictions served: {portfolio_event['total_predictions']}. "
                + (
                    f"Best performing project: {portfolio_event['best_performer']['name']} "
                    f"({portfolio_event['best_performer']['algorithm']}, "
                    f"{int((portfolio_event['best_performer']['metric_value'] or 0) * 100)}% "
                    f"{portfolio_event['best_performer']['metric_name']})."
                    if portfolio_event.get("best_performer")
                    else "No trained models yet."
                )
                + " Present the portfolio overview from the card. "
                "Help the analyst understand which projects are performing best "
                "and what actions they might take next."
            )
        except Exception:  # noqa: BLE001
            pass  # Portfolio is nice-to-have; never crash chat

    # Rate limit / quota management
    rate_limit_event: dict | None = None
    if _RATE_LIMIT_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            _dep = ctx["deployment"]
            _dep_id = _dep.id if hasattr(_dep, "id") else str(_dep)
            _rpm: int | None = None
            _quota: int | None = None
            _disable_rpm = bool(_DISABLE_RATE_RE.search(body.message))
            _disable_quota = bool(_DISABLE_QUOTA_RE.search(body.message))

            _rpm_m = _RATE_LIMIT_NUMBER_RE.search(body.message)
            if _rpm_m:
                _rpm = int(_rpm_m.group(1))
            _quota_m = _QUOTA_NUMBER_RE.search(body.message)
            if _quota_m:
                _quota = int(_quota_m.group(1))

            # Check if this is just a status check (no set/disable)
            _status_only = not (_rpm or _quota or _disable_rpm or _disable_quota)
            # "quota status", "check quota", "how many predictions left"
            _status_query = bool(
                re.search(
                    r"\b(quota\s+status|check\s+quota|how\s+many\s+predictions?\s+(?:left|remaining)|usage\s+stats)\b",
                    body.message,
                    re.IGNORECASE,
                )
            )

            if not _status_only or _status_query:
                if _disable_rpm:
                    _rpm = 0  # 0 = remove limit
                if _disable_quota:
                    _quota = 0  # 0 = remove quota

                # Apply changes if any were requested
                if _rpm is not None or _quota is not None:
                    _dep_obj = session.get(Deployment, _dep_id)
                    if _dep_obj:
                        if _rpm is not None:
                            _dep_obj.rate_limit_rpm = _rpm if _rpm > 0 else None
                        if _quota is not None:
                            _dep_obj.monthly_quota = _quota if _quota > 0 else None
                        session.add(_dep_obj)
                        session.commit()
                        session.refresh(_dep_obj)
                        _dep = _dep_obj

            # Always compute current quota usage for the event
            _current_rpm = getattr(_dep, "rate_limit_rpm", None)
            _current_quota = getattr(_dep, "monthly_quota", None)

            from datetime import timedelta as _td
            from sqlmodel import func as _sqlfunc

            _cutoff = _utcnow() - _td(days=30)
            _used = session.exec(
                select(_sqlfunc.count(PredictionLog.id)).where(
                    PredictionLog.deployment_id == _dep_id,
                    PredictionLog.created_at >= _cutoff,
                )
            ).one()

            _remaining = (_current_quota - _used) if _current_quota else None
            _pct = round(_used / _current_quota * 100, 1) if _current_quota else None

            rate_limit_event = {
                "deployment_id": _dep_id,
                "rate_limit_rpm": _current_rpm,
                "rate_limit_enabled": _current_rpm is not None,
                "monthly_quota": _current_quota,
                "quota_enabled": _current_quota is not None,
                "used_this_month": _used,
                "remaining": _remaining,
                "pct_used": _pct,
                "summary": (
                    (
                        f"Rate limit: {_current_rpm} req/min. "
                        if _current_rpm
                        else "No per-minute rate limit. "
                    )
                    + (
                        f"Monthly quota: {_used}/{_current_quota} predictions used "
                        f"({_pct}% — {_remaining} remaining)."
                        if _current_quota
                        else "No monthly quota configured."
                    )
                ),
            }
            system_prompt += (
                f"\n\n## Rate Limit Configuration\n"
                f"{rate_limit_event['summary']} "
                "Tell the analyst the current rate limit and quota status in plain English. "
                "If limits were just changed, confirm the new settings and explain what they mean. "
                "If checking status, summarise the quota usage and whether they are close to the limit."
            )
        except Exception:  # noqa: BLE001
            pass  # Rate limit events are nice-to-have; never crash chat

    # SLA / latency monitoring
    sla_metrics_event: dict | None = None
    if _SLA_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            _dep = ctx["deployment"]
            _dep_id = _dep.id if hasattr(_dep, "id") else str(_dep)

            from collections import defaultdict as _defaultdict

            _logs = session.exec(
                select(PredictionLog).where(PredictionLog.deployment_id == _dep_id)
            ).all()
            _timed = [lg for lg in _logs if lg.response_ms is not None]

            if not _timed:
                sla_metrics_event = {
                    "deployment_id": _dep_id,
                    "sample_count": 0,
                    "p50_ms": None,
                    "p95_ms": None,
                    "p99_ms": None,
                    "avg_ms": None,
                    "alert": False,
                    "alert_message": None,
                    "latency_by_day": [],
                    "summary": "No timing data yet — latency will appear after the first prediction.",
                }
            else:
                from api.deploy import _percentile as _pctile

                _lats = sorted(lg.response_ms for lg in _timed)  # type: ignore[misc]
                _p50 = _pctile(_lats, 50)
                _p95 = _pctile(_lats, 95)
                _p99 = _pctile(_lats, 99)
                _avg = round(sum(_lats) / len(_lats), 2)
                _alert = _p95 > 500.0
                _alert_msg = (
                    f"p95 latency is {_p95}ms — above the 500ms target. "
                    "Consider retraining with fewer features or switching to a simpler algorithm."
                    if _alert
                    else None
                )

                _day_totals: dict = _defaultdict(list)
                for _lg in _timed:
                    _day_totals[_lg.created_at.strftime("%Y-%m-%d")].append(
                        _lg.response_ms
                    )

                _by_day = [
                    {"date": d, "avg_ms": round(sum(ms) / len(ms), 2)}
                    for d, ms in sorted(_day_totals.items())
                ]

                _status = "Alert" if _alert else "Healthy"
                sla_metrics_event = {
                    "deployment_id": _dep_id,
                    "sample_count": len(_timed),
                    "p50_ms": _p50,
                    "p95_ms": _p95,
                    "p99_ms": _p99,
                    "avg_ms": _avg,
                    "alert": _alert,
                    "alert_message": _alert_msg,
                    "latency_by_day": _by_day,
                    "summary": (
                        f"Prediction latency ({_status}): p50={_p50}ms, p95={_p95}ms, p99={_p99}ms "
                        f"— based on {len(_timed)} timed prediction{'s' if len(_timed) != 1 else ''}."
                    ),
                }
            system_prompt += (
                f"\n\n## Prediction Latency\n"
                f"{sla_metrics_event['summary']} "
                "Explain prediction latency in plain English. If there is an alert, "
                "tell the analyst whether their p95 latency is above 500ms and suggest "
                "switching to a simpler algorithm or reducing features to improve speed. "
                "If the latency is healthy, reassure them."
            )
        except Exception:  # noqa: BLE001
            pass  # SLA events are nice-to-have; never crash chat

    # Quota alert threshold configuration
    quota_alert_event: dict | None = None
    if _QUOTA_ALERT_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            _dep = ctx["deployment"]
            _dep_id = _dep.id if hasattr(_dep, "id") else str(_dep)
            _dep_obj = session.get(Deployment, _dep_id)

            _disable_alert = bool(_DISABLE_QUOTA_ALERT_RE.search(body.message))
            _alert_pct_m = _QUOTA_ALERT_PCT_RE.search(body.message)
            _new_alert_pct: int | None = (
                int(_alert_pct_m.group(1)) if _alert_pct_m else None
            )

            if _dep_obj:
                if _disable_alert:
                    _dep_obj.quota_alert_threshold_pct = None
                    session.add(_dep_obj)
                    session.commit()
                    session.refresh(_dep_obj)
                elif _new_alert_pct is not None and 1 <= _new_alert_pct <= 99:
                    _dep_obj.quota_alert_threshold_pct = _new_alert_pct
                    session.add(_dep_obj)
                    session.commit()
                    session.refresh(_dep_obj)

                _threshold = getattr(_dep_obj, "quota_alert_threshold_pct", None)
                _monthly_quota = getattr(_dep_obj, "monthly_quota", None)

                from datetime import timedelta as _td2
                from sqlmodel import func as _sqlfunc2

                _cutoff2 = _utcnow() - _td2(days=30)
                _used2 = session.exec(
                    select(_sqlfunc2.count(PredictionLog.id)).where(
                        PredictionLog.deployment_id == _dep_id,
                        PredictionLog.created_at >= _cutoff2,
                    )
                ).one()
                _pct2 = (
                    round(_used2 / _monthly_quota * 100, 1) if _monthly_quota else None
                )

                quota_alert_event = {
                    "deployment_id": _dep_id,
                    "quota_alert_enabled": _threshold is not None,
                    "quota_alert_threshold_pct": _threshold,
                    "monthly_quota": _monthly_quota,
                    "used_this_month": _used2,
                    "pct_used": _pct2,
                    "summary": (
                        f"Quota alert set at {_threshold}% of {_monthly_quota} predictions "
                        f"(currently at {_pct2}%)."
                        if _threshold and _monthly_quota
                        else (
                            f"Quota alert set at {_threshold}% — configure a monthly quota to activate."
                            if _threshold
                            else "Quota alerts are disabled."
                        )
                    ),
                }
                system_prompt += (
                    f"\n\n## Quota Alert Configuration\n"
                    f"{quota_alert_event['summary']} "
                    "Tell the analyst the current quota alert setting in plain English. "
                    "If a threshold was just set, confirm it and explain that they will receive "
                    "a webhook notification when usage reaches that percentage of their monthly quota. "
                    "If alerts were disabled, confirm that. "
                    "Remind them that webhooks must be configured to receive notifications."
                )
        except Exception:  # noqa: BLE001
            pass  # Quota alert events are nice-to-have; never crash chat

    # A/B test status / promote / end
    ab_test_result_event: dict | None = None
    if _AB_TEST_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            from models.ab_test import ABTest as _ABTest

            _ab_dep = ctx["deployment"]
            _ab_dep_id = _ab_dep.id if hasattr(_ab_dep, "id") else str(_ab_dep)

            _active_test = session.exec(
                select(_ABTest).where(
                    _ABTest.champion_id == _ab_dep_id,
                    _ABTest.is_active == True,  # noqa: E712
                )
            ).first()

            if _AB_PROMOTE_RE.search(body.message) and _active_test:
                # Promote challenger — replicate promote_challenger() logic
                from models.deployment import Deployment as _Dep2
                from models.deployment_version import DeploymentVersion as _DV

                _champ = session.get(_Dep2, _ab_dep_id)
                _chall = session.get(_Dep2, _active_test.challenger_id)
                if _champ and _chall and _chall.pipeline_path:
                    from api.deploy import _archive_current_version

                    _archive_current_version(_champ, session)
                    _new_ver = getattr(_champ, "current_version_number", 1) + 1
                    _champ.model_run_id = _chall.model_run_id
                    _champ.pipeline_path = _chall.pipeline_path
                    _champ.algorithm = _chall.algorithm
                    _champ.problem_type = _chall.problem_type
                    _champ.feature_names = _chall.feature_names
                    _champ.target_column = _chall.target_column
                    _champ.metrics = _chall.metrics
                    _champ.current_version_number = _new_ver
                    session.add(_champ)
                    session.add(
                        _DV(
                            deployment_id=_ab_dep_id,
                            version_number=_new_ver,
                            model_run_id=_chall.model_run_id,
                            algorithm=_chall.algorithm,
                            problem_type=_chall.problem_type,
                            target_column=_chall.target_column,
                            metrics=_chall.metrics,
                            pipeline_path=_chall.pipeline_path,
                            is_current=True,
                        )
                    )
                    _active_test.is_active = False
                    _active_test.ended_at = datetime.now(UTC).replace(tzinfo=None)
                    _active_test.winner = "challenger"
                    session.add(_active_test)
                    session.commit()
                    ab_test_result_event = {
                        "action": "promoted",
                        "summary": (
                            f"Challenger ({_chall.algorithm}) promoted to champion. "
                            "Your prediction URL stays the same."
                        ),
                    }
                    system_prompt += (
                        "\n\n## A/B Test Outcome\n"
                        f"The challenger model ({_chall.algorithm}) has been promoted to "
                        "champion. The prediction endpoint URL is unchanged so any VP or "
                        "developer links continue to work."
                    )
                else:
                    ab_test_result_event = {
                        "action": "none",
                        "summary": "Promotion failed: challenger pipeline not found.",
                    }

            elif _AB_END_RE.search(body.message) and _active_test:
                # End the A/B test without promoting
                _active_test.is_active = False
                _active_test.ended_at = datetime.now(UTC).replace(tzinfo=None)
                session.add(_active_test)
                session.commit()
                ab_test_result_event = {
                    "action": "ended",
                    "summary": "A/B test ended. Champion model remains active.",
                }
                system_prompt += (
                    "\n\n## A/B Test Ended\n"
                    "The A/B test has been stopped. The original champion model remains "
                    "active and continues to handle all prediction traffic."
                )

            elif _active_test:
                # Status report — build full response via helper
                from api.deploy import _ab_test_response as _ab_resp

                _ab_data = _ab_resp(_active_test, session)
                ab_test_result_event = {
                    "action": "status",
                    "summary": (
                        f"A/B test active: {_ab_data['champion_split_pct']}% champion "
                        f"({_ab_data['champion_algorithm'] or 'Model'}) / "
                        f"{_ab_data['challenger_split_pct']}% challenger "
                        f"({_ab_data['challenger_algorithm'] or 'Model'}). "
                        f"Champion: {_ab_data['champion_metrics']['request_count']} requests. "
                        f"Challenger: {_ab_data['challenger_metrics']['request_count']} requests."
                    ),
                    **_ab_data,
                }
                system_prompt += (
                    f"\n\n## Active A/B Test\n"
                    f"Traffic split: {_ab_data['champion_split_pct']}% champion / "
                    f"{_ab_data['challenger_split_pct']}% challenger. "
                    f"Champion requests: {_ab_data['champion_metrics']['request_count']}, "
                    f"Challenger requests: {_ab_data['challenger_metrics']['request_count']}. "
                    f"Statistical significance: {_ab_data['significance']['note']}. "
                    "Tell the analyst what the test data shows in plain English. "
                    "If there are enough samples and a result, recommend whether to promote "
                    "or keep running."
                )

            else:
                # No active test
                ab_test_result_event = {
                    "action": "none",
                    "summary": (
                        "No active A/B test. You can start one from the Deployment panel "
                        "once you have a second trained model as a challenger."
                    ),
                }
                system_prompt += (
                    "\n\n## A/B Test\nNo active A/B test is running for this deployment. "
                    "Explain to the analyst how to start one: train a second model, deploy it, "
                    "then use the Deployment panel A/B Test section to split traffic."
                )

        except Exception:  # noqa: BLE001
            pass  # A/B test events are nice-to-have; never crash chat

    # Webhook event history — "what webhooks fired recently?"
    webhook_history_event: dict | None = None
    if _WEBHOOK_HISTORY_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            from models.webhook_event import WebhookEvent as _WHEvt
            from models.webhook_config import WebhookConfig as _WHCfg

            _wh_dep = ctx["deployment"]
            _wh_dep_id = _wh_dep.id if hasattr(_wh_dep, "id") else str(_wh_dep)

            _wh_events = session.exec(
                select(_WHEvt)
                .where(_WHEvt.deployment_id == _wh_dep_id)
                .order_by(_WHEvt.fired_at.desc())
                .limit(10)
            ).all()

            # Look up webhook URLs
            _wh_hooks = session.exec(
                select(_WHCfg).where(_WHCfg.deployment_id == _wh_dep_id)
            ).all()
            _url_map: dict[str, str] = {h.id: h.url for h in _wh_hooks}

            _wh_event_list = [
                {
                    "id": e.id,
                    "webhook_id": e.webhook_id,
                    "webhook_url": _url_map.get(e.webhook_id, "(deleted)"),
                    "event_type": e.event_type,
                    "fired_at": e.fired_at.isoformat() if e.fired_at else None,
                    "status_code": e.status_code,
                    "success": (
                        (200 <= (e.status_code or 0) < 300) if e.status_code else False
                    ),
                }
                for e in _wh_events
            ]

            _wh_total = len(_wh_event_list)
            if _wh_total == 0:
                _wh_summary = "No webhook events have fired for this deployment yet."
            else:
                _wh_successes = sum(1 for e in _wh_event_list if e["success"])
                _wh_types = list(dict.fromkeys(e["event_type"] for e in _wh_event_list))
                _wh_summary = (
                    f"{_wh_total} recent webhook event{'s' if _wh_total != 1 else ''} "
                    f"({_wh_successes} successful). "
                    f"Event types seen: {', '.join(_wh_types[:3])}."
                )

            webhook_history_event = {
                "total": _wh_total,
                "events": _wh_event_list,
                "summary": _wh_summary,
            }
            system_prompt += (
                f"\n\n## Webhook Event History\n"
                f"{_wh_summary} "
                f"Present this as a brief timeline. "
                f"If no events: explain that webhooks fire when events like batch completion, "
                f"drift detection, health degradation, or quota alerts occur."
            )

        except Exception:  # noqa: BLE001
            pass  # Webhook history is nice-to-have; never crash chat

    # Batch prediction schedule creation / listing
    schedule_event: dict | None = None
    if _SCHEDULE_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            _sched_dep = ctx["deployment"]
            _sched_dep_id = (
                _sched_dep.id if hasattr(_sched_dep, "id") else str(_sched_dep)
            )

            _is_list_request = bool(
                re.search(r"\b(show|list|view|what|get)\s", body.message, re.IGNORECASE)
            )

            if _is_list_request:
                from models.batch_schedule import BatchSchedule as _BS2

                _existing = session.exec(
                    select(_BS2).where(_BS2.deployment_id == _sched_dep_id)
                ).all()
                _sched_list = [
                    {
                        "id": s.id,
                        "frequency": s.frequency,
                        "run_hour": s.run_hour,
                        "run_minute": s.run_minute,
                        "day_of_week": s.day_of_week,
                        "day_of_month": s.day_of_month,
                        "next_run": s.next_run.isoformat() if s.next_run else None,
                        "last_run": s.last_run.isoformat() if s.last_run else None,
                        "last_row_count": s.last_row_count,
                        "description": _build_schedule_description(
                            s.frequency,
                            s.run_hour,
                            s.run_minute,
                            s.day_of_week,
                            s.day_of_month,
                        ),
                    }
                    for s in _existing
                ]
                schedule_event = {
                    "action": "list",
                    "deployment_id": _sched_dep_id,
                    "schedules": _sched_list,
                    "count": len(_sched_list),
                    "summary": (
                        f"You have {len(_sched_list)} batch schedule(s)."
                        if _sched_list
                        else "No batch schedules configured yet."
                    ),
                }
                system_prompt += (
                    f"\n\n## Batch Schedules\n{schedule_event['summary']} "
                    "List the schedules in plain English and tell the analyst they can "
                    "manage schedules in the Deployment panel."
                )
            else:
                # Create a new schedule
                _params = _extract_schedule_params(body.message)
                from core.scheduler import compute_next_run as _cnr2
                from models.batch_schedule import BatchSchedule as _BS3

                _new_sched = _BS3(
                    deployment_id=_sched_dep_id,
                    frequency=_params["frequency"],
                    run_hour=_params["run_hour"],
                    run_minute=_params["run_minute"],
                    day_of_week=_params["day_of_week"],
                    day_of_month=_params["day_of_month"],
                )
                _new_sched.next_run = _cnr2(
                    _params["frequency"],
                    _params["run_hour"],
                    _params["run_minute"],
                    _params["day_of_week"],
                    _params["day_of_month"],
                )
                session.add(_new_sched)
                session.commit()
                session.refresh(_new_sched)

                _desc = _build_schedule_description(
                    _params["frequency"],
                    _params["run_hour"],
                    _params["run_minute"],
                    _params["day_of_week"],
                    _params["day_of_month"],
                )
                schedule_event = {
                    "action": "created",
                    "schedule_id": _new_sched.id,
                    "deployment_id": _sched_dep_id,
                    "frequency": _params["frequency"],
                    "run_hour": _params["run_hour"],
                    "run_minute": _params["run_minute"],
                    "day_of_week": _params["day_of_week"],
                    "day_of_month": _params["day_of_month"],
                    "next_run": (
                        _new_sched.next_run.isoformat() if _new_sched.next_run else None
                    ),
                    "description": _desc,
                    "summary": f"Batch predictions scheduled: {_desc}.",
                }
                system_prompt += (
                    f"\n\n## Batch Schedule Created\n{_desc}. "
                    "Tell the analyst their batch prediction schedule has been set up. "
                    "Explain that the model will automatically score their current dataset "
                    "on this schedule and save results as a downloadable CSV. "
                    "Mention they can view and manage schedules in the Deployment panel."
                )
        except Exception:  # noqa: BLE001
            pass  # Schedule events are nice-to-have; never crash chat

    # Class imbalance detection via chat
    class_imbalance_event: dict | None = None
    if _CLASS_IMBALANCE_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ci_dataset = ctx["dataset"]
            _ci_fs = ctx["feature_set"]
            _ci_file = getattr(_ci_dataset, "file_path", None)
            _ci_target = None
            if _ci_fs:
                _ci_target = getattr(_ci_fs, "target_column", None)

            if _ci_file and _ci_target and Path(_ci_file).exists():
                from core.trainer import detect_class_imbalance as _detect_imbalance

                _ci_df = pd.read_csv(_ci_file)
                if _ci_target in _ci_df.columns:
                    _ci_problem_type = (
                        getattr(_ci_fs, "problem_type", None) or "unknown"
                    )
                    if _ci_problem_type == "classification":
                        _ci_y = _ci_df[_ci_target].dropna().astype(str).values
                        _ci_result = _detect_imbalance(_ci_y)
                        _ci_result["project_id"] = project_id
                        _ci_result["target_column"] = _ci_target
                        _ci_result["problem_type"] = _ci_problem_type
                        class_imbalance_event = _ci_result
                        _ci_pct = round(
                            (_ci_result.get("minority_ratio") or 0) * 100, 1
                        )
                        system_prompt += (
                            f"\n\n## Class Imbalance Analysis\n"
                            f"Target: {_ci_target} | Problem type: {_ci_problem_type}\n"
                            + (
                                f"Imbalanced: yes — minority class is {_ci_pct}% of rows. "
                                f"Recommended strategy: {_ci_result['recommended_strategy']}. "
                                f"{_ci_result['explanation']}\n\n"
                                "Explain class imbalance in plain English: the model will be "
                                "biased toward the majority class and will miss the rare "
                                "cases that often matter most (fraud, churn, failures). "
                                f"Recommend the {_ci_result['recommended_strategy']} strategy "
                                "and explain how to apply it via the Models tab."
                                if _ci_result["is_imbalanced"]
                                else f"Balanced: classes are well-distributed. "
                                f"{_ci_result['explanation']}\n\n"
                                "Reassure the analyst their target classes are balanced — "
                                "no special handling is needed before training."
                            )
                        )
                    else:
                        # Regression — class imbalance doesn't apply
                        class_imbalance_event = {
                            "project_id": project_id,
                            "target_column": _ci_target,
                            "problem_type": _ci_problem_type,
                            "is_imbalanced": False,
                            "class_distribution": [],
                            "minority_class": None,
                            "minority_ratio": None,
                            "recommended_strategy": "none",
                            "explanation": (
                                "Class imbalance only applies to classification problems. "
                                "Your target column is numeric (regression) — no balancing needed."
                            ),
                        }
                        system_prompt += (
                            "\n\n## Class Imbalance\n"
                            "Class imbalance does not apply to regression problems. "
                            "Explain that the target column is continuous, so class balancing "
                            "strategies are not relevant here."
                        )
        except Exception:  # noqa: BLE001
            pass  # Nice-to-have; never crash chat

    # Pre-compute follow-up suggestions (based on state + current message)
    current_state = detect_state(
        ctx["dataset"], ctx["feature_set"], ctx["model_runs"], ctx["deployment"]
    )
    suggestions_list = generate_suggestions(
        state=current_state,
        dataset=ctx["dataset"],
        feature_set=ctx["feature_set"],
        model_runs=ctx["model_runs"],
        deployment=ctx["deployment"],
        last_user_message=body.message,
    )

    def stream_response():
        full_response = ""
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    chunk = json.dumps({"type": "token", "content": text})
                    yield f"data: {chunk}\n\n"

        finally:
            # Save assistant response
            messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": _utcnow().isoformat(),
                }
            )
            from db import engine

            with Session(engine) as save_session:
                conv = save_session.get(Conversation, conversation.id)
                if conv:
                    conv.messages = json.dumps(messages)
                    conv.updated_at = _utcnow()
                    save_session.add(conv)
                    save_session.commit()

        # Emit readiness card if computed
        if readiness_data:
            yield f"data: {json.dumps({'type': 'readiness', 'readiness': readiness_data})}\n\n"

        # Emit drift card if computed
        if drift_data:
            yield f"data: {json.dumps({'type': 'drift', 'drift': drift_data})}\n\n"

        # Emit tune suggestion if detected
        if tune_data:
            yield f"data: {json.dumps({'type': 'tune', 'tune': tune_data})}\n\n"

        # Emit model improvement suggestions if computed
        if improvement_event:
            yield f"data: {json.dumps({'type': 'model_improvement', 'model_improvement': improvement_event})}\n\n"

        # Emit model selection recommendation if computed
        if model_select_event:
            yield f"data: {json.dumps({'type': 'model_selection', 'model_selection': model_select_event})}\n\n"

        # Emit goal-driven training result if computed
        if goal_train_event:
            yield f"data: {json.dumps({'type': 'goal_training', 'goal_training': goal_train_event})}\n\n"

        if auto_retrain_event:
            yield f"data: {json.dumps({'type': 'auto_retrain', 'auto_retrain': auto_retrain_event})}\n\n"

        if conv_export_event:
            yield f"data: {json.dumps({'type': 'conversation_export', 'conversation_export': conv_export_event})}\n\n"

        if health_summary_event:
            yield f"data: {json.dumps({'type': 'health_summary', 'health_summary': health_summary_event})}\n\n"

        if predict_opp_event:
            yield f"data: {json.dumps({'type': 'prediction_opportunities', 'prediction_opportunities': predict_opp_event})}\n\n"

        if dataset_compare_event:
            yield f"data: {json.dumps({'type': 'dataset_comparison', 'dataset_comparison': dataset_compare_event})}\n\n"

        if version_history_event:
            yield f"data: {json.dumps({'type': 'version_history', 'version_history': version_history_event})}\n\n"

        # Emit model health card if computed
        if health_data:
            yield f"data: {json.dumps({'type': 'health', 'health': health_data})}\n\n"

        # Emit deployment alerts if scanned
        if alerts_data:
            yield f"data: {json.dumps({'type': 'alerts', 'alerts': alerts_data})}\n\n"

        # Emit model history trigger if detected
        if history_event:
            yield f"data: {json.dumps({'type': 'history', 'history': history_event})}\n\n"

        # Emit analytics trigger if detected
        if analytics_event:
            yield f"data: {json.dumps({'type': 'analytics', 'analytics': analytics_event})}\n\n"

        # Emit cross-tabulation / pivot table if computed
        if crosstab_event:
            yield f"data: {json.dumps({'type': 'crosstab', 'crosstab': crosstab_event})}\n\n"

        # Emit anomaly detection results if computed
        if anomaly_event:
            yield f"data: {json.dumps({'type': 'anomalies', 'anomalies': anomaly_event})}\n\n"

        # Emit cleaning suggestion (user must click to apply — "explain before executing")
        if cleaning_suggestion:
            yield f"data: {json.dumps({'type': 'cleaning_suggestion', 'cleaning': cleaning_suggestion})}\n\n"

        # Emit computed column suggestion (user must click Apply — "explain before executing")
        if compute_suggestion:
            yield f"data: {json.dumps({'type': 'compute_suggestion', 'compute': compute_suggestion})}\n\n"

        # Emit segment comparison table if computed
        if segment_comparison_event:
            yield f"data: {json.dumps({'type': 'segment_comparison', 'segment_comparison': segment_comparison_event})}\n\n"

        # Emit data readiness assessment
        if data_readiness_event:
            yield f"data: {json.dumps({'type': 'data_readiness', 'readiness': data_readiness_event})}\n\n"

        # Emit target correlation analysis
        if target_correlation_event:
            yield f"data: {json.dumps({'type': 'target_correlation', 'correlation': target_correlation_event})}\n\n"

        # Emit group-by analysis
        if group_stats_event:
            yield f"data: {json.dumps({'type': 'group_stats', 'group_stats': group_stats_event})}\n\n"

        # Emit forecast chart if computed
        if forecast_event:
            yield f"data: {json.dumps({'type': 'forecast', 'forecast': forecast_event})}\n\n"

        # Emit refresh prompt — guides user to upload new data
        if refresh_prompt_event:
            yield f"data: {json.dumps({'type': 'refresh_prompt', 'refresh': refresh_prompt_event})}\n\n"

        # Emit feature selection analysis — ranked importances + list of weak features
        if feature_sel_event:
            yield f"data: {json.dumps({'type': 'feature_selection', 'feature_selection': feature_sel_event})}\n\n"

        # Emit split strategy event — user changed split method preference
        if split_strategy_event:
            yield f"data: {json.dumps({'type': 'split_strategy', 'split_strategy': split_strategy_event})}\n\n"

        # Emit training started event — backend has already launched training threads
        if training_started_event:
            yield f"data: {json.dumps({'type': 'training_started', 'training': training_started_event})}\n\n"

        # Emit model card — plain-English model explanation
        if model_card_event:
            yield f"data: {json.dumps({'type': 'model_card', 'model_card': model_card_event})}\n\n"

        # Emit PDF report ready event — provides download URL
        if report_ready_event:
            yield f"data: {json.dumps({'type': 'report_ready', 'report': report_ready_event})}\n\n"

        # Emit feature engineering suggestions card
        if feature_suggestions_event:
            yield f"data: {json.dumps({'type': 'feature_suggestions', 'suggestions': feature_suggestions_event})}\n\n"

        # Emit features applied confirmation card
        if features_applied_event:
            yield f"data: {json.dumps({'type': 'features_applied', 'applied': features_applied_event})}\n\n"
            yield f"data: {json.dumps({'type': 'next_step', 'chips': get_next_step_chips('shape')})}\n\n"

        # Emit deployed event — model is now live
        if deployed_event:
            yield f"data: {json.dumps({'type': 'deployed', 'deployment': deployed_event})}\n\n"
            yield f"data: {json.dumps({'type': 'next_step', 'chips': get_next_step_chips('deploy')})}\n\n"

        # Emit automated data story
        if data_story_event:
            yield f"data: {json.dumps({'type': 'data_story', 'story': data_story_event})}\n\n"

        # Emit filter set event (filter is now active on the dataset)
        if filter_set_event:
            yield f"data: {json.dumps({'type': 'filter_set', 'filter': filter_set_event})}\n\n"

        # Emit filter cleared event (returning to full dataset)
        if filter_cleared_event:
            yield f"data: {json.dumps({'type': 'filter_cleared', 'filter': filter_cleared_event})}\n\n"

        # Emit segment performance breakdown (how does model perform per segment?)
        if segment_performance_event:
            yield f"data: {json.dumps({'type': 'segment_performance', 'segment_performance': segment_performance_event})}\n\n"

        # Emit column profile deep-dive (tell me about the revenue column)
        if column_profile_event:
            yield f"data: {json.dumps({'type': 'column_profile', 'column_profile': column_profile_event})}\n\n"

        # Emit K-means clustering result
        if cluster_event:
            yield f"data: {json.dumps({'type': 'clusters', 'clusters': cluster_event})}\n\n"

        # Emit top-N ranking result
        if top_n_event:
            yield f"data: {json.dumps({'type': 'top_n', 'top_n': top_n_event})}\n\n"

        # Emit prediction error analysis result
        if pred_error_event:
            yield f"data: {json.dumps({'type': 'prediction_errors', 'pred_errors': pred_error_event})}\n\n"

        # Emit onboarding wizard card
        if onboarding_event:
            yield f"data: {json.dumps({'type': 'onboarding_guide', 'onboarding_guide': onboarding_event})}\n\n"

        # Emit record table viewer result
        if records_event:
            yield f"data: {json.dumps({'type': 'records', 'records': records_event})}\n\n"

        # Emit what-if prediction result
        if whatif_chat_event:
            yield f"data: {json.dumps({'type': 'whatif_result', 'whatif': whatif_chat_event})}\n\n"

        # Emit multi-row batch prediction result (multiple scenarios in one message)
        if multi_pred_event:
            yield f"data: {json.dumps({'type': 'multi_prediction', 'multi_prediction': multi_pred_event})}\n\n"

        # Emit inline multi-feature prediction result
        if inline_pred_event:
            yield f"data: {json.dumps({'type': 'inline_prediction', 'inline_prediction': inline_pred_event})}\n\n"

        # Emit sensitivity analysis result
        if sensitivity_event:
            yield f"data: {json.dumps({'type': 'sensitivity', 'sensitivity': sensitivity_event})}\n\n"

        # Emit feature interaction heatmap result
        if interaction_event:
            yield f"data: {json.dumps({'type': 'interaction', 'interaction': interaction_event})}\n\n"

        # Emit dataset ranking result
        if ranked_pred_event:
            yield f"data: {json.dumps({'type': 'ranked_predictions', 'ranked_predictions': ranked_pred_event})}\n\n"

        # Emit prediction cohort profile result
        if cohort_event:
            yield f"data: {json.dumps({'type': 'prediction_cohort', 'prediction_cohort': cohort_event})}\n\n"

        # Emit partial dependence plot result
        if pdp_event:
            yield f"data: {json.dumps({'type': 'partial_dependence', 'partial_dependence': pdp_event})}\n\n"

        # Emit calibration check result
        if calibration_check_event:
            yield f"data: {json.dumps({'type': 'calibration_check', 'calibration_check': calibration_check_event})}\n\n"

        # Emit learning curve analysis result
        if learning_curve_event:
            yield f"data: {json.dumps({'type': 'learning_curve', 'learning_curve': learning_curve_event})}\n\n"

        # Emit time-period comparison result
        if time_window_event:
            yield f"data: {json.dumps({'type': 'time_window_comparison', 'time_window': time_window_event})}\n\n"

        # Emit follow-up suggestion chips (always, if we have any)
        if suggestions_list:
            yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions_list})}\n\n"

        # Emit correlation heatmap if triggered (reuses existing {type:"chart"} path)
        if heatmap_chart:
            yield f"data: {json.dumps({'type': 'chart', 'chart': heatmap_chart})}\n\n"

        # Emit scatter chart if triggered (reuses existing {type:"chart"} path)
        if scatter_chart:
            yield f"data: {json.dumps({'type': 'chart', 'chart': scatter_chart})}\n\n"

        # Emit line/trend chart if triggered (reuses existing {type:"chart"} path)
        if line_chart:
            yield f"data: {json.dumps({'type': 'chart', 'chart': line_chart})}\n\n"

        # Emit box plot chart if triggered (reuses existing {type:"chart"} path)
        if boxplot_chart:
            yield f"data: {json.dumps({'type': 'chart', 'chart': boxplot_chart})}\n\n"

        # Emit pie / donut chart if triggered (reuses existing {type:"chart"} path)
        if pie_chart:
            yield f"data: {json.dumps({'type': 'chart', 'chart': pie_chart})}\n\n"

        # Emit bar chart if triggered (reuses existing {type:"chart"} path)
        if bar_chart:
            yield f"data: {json.dumps({'type': 'chart', 'chart': bar_chart})}\n\n"

        # Emit histogram chart if triggered (reuses existing {type:"chart"} path)
        if histogram_chart:
            yield f"data: {json.dumps({'type': 'chart', 'chart': histogram_chart})}\n\n"

        # Emit missing values overview card
        if null_map_event:
            yield f"data: {json.dumps({'type': 'null_map', 'null_map': null_map_event})}\n\n"

        # Emit summary statistics table
        if summary_stats_event:
            yield f"data: {json.dumps({'type': 'summary_stats', 'summary_stats': summary_stats_event})}\n\n"

        # Emit value counts table
        if value_counts_event:
            yield f"data: {json.dumps({'type': 'value_counts', 'value_counts': value_counts_event})}\n\n"

        # Emit pair correlation card
        if pair_corr_event:
            yield f"data: {json.dumps({'type': 'pair_correlation', 'pair_correlation': pair_corr_event})}\n\n"

        # Emit stat query card
        if stat_query_event:
            yield f"data: {json.dumps({'type': 'stat_query', 'stat_query': stat_query_event})}\n\n"

        # Emit group trend analysis card
        if group_trends_event:
            yield f"data: {json.dumps({'type': 'group_trends', 'group_trends': group_trends_event})}\n\n"

        # Emit dataset export card if triggered
        if data_export:
            yield f"data: {json.dumps({'type': 'data_export', 'data_export': data_export})}\n\n"

        # Emit column rename result if executed
        if rename_result:
            yield f"data: {json.dumps({'type': 'rename_result', 'rename': rename_result})}\n\n"

        # Emit analysis template saved confirmation
        if template_saved_event:
            yield f"data: {json.dumps({'type': 'template_saved', 'template': template_saved_event})}\n\n"

        # Emit analysis template list
        if template_list_event:
            yield f"data: {json.dumps({'type': 'template_list', 'template_list': template_list_event})}\n\n"

        # Emit analysis template replay — queries shown as clickable chips
        if template_replay_event:
            yield f"data: {json.dumps({'type': 'template_replay', 'template_replay': template_replay_event})}\n\n"

        # Emit prediction preset saved confirmation
        if preset_saved_event:
            yield f"data: {json.dumps({'type': 'preset_saved', 'preset': preset_saved_event})}\n\n"

        # Emit prediction preset list
        if preset_list_event:
            yield f"data: {json.dumps({'type': 'preset_list', 'preset_list': preset_list_event})}\n\n"

        # Emit SDK download card
        if sdk_event:
            yield f"data: {json.dumps({'type': 'sdk_download', 'sdk_download': sdk_event})}\n\n"

        # Emit cross-project portfolio overview
        if portfolio_event:
            yield f"data: {json.dumps({'type': 'portfolio', 'portfolio': portfolio_event})}\n\n"

        # Emit rate limit / quota status card
        if rate_limit_event:
            yield f"data: {json.dumps({'type': 'rate_limit', 'rate_limit': rate_limit_event})}\n\n"

        # Emit SLA / latency metrics card
        if sla_metrics_event:
            yield f"data: {json.dumps({'type': 'sla_metrics', 'sla_metrics': sla_metrics_event})}\n\n"

        # Emit quota alert configuration card
        if quota_alert_event:
            yield f"data: {json.dumps({'type': 'quota_alert_config', 'quota_alert_config': quota_alert_event})}\n\n"

        # Emit batch prediction schedule event
        if schedule_event:
            yield f"data: {json.dumps({'type': 'schedule_set', 'schedule_set': schedule_event})}\n\n"

        # Emit webhook event history
        if webhook_history_event:
            yield f"data: {json.dumps({'type': 'webhook_history', 'webhook_history': webhook_history_event})}\n\n"

        # Emit A/B test status / action confirmation
        if ab_test_result_event:
            yield f"data: {json.dumps({'type': 'ab_test_result', 'ab_test_result': ab_test_result_event})}\n\n"

        # Emit class imbalance detection result
        if class_imbalance_event:
            yield f"data: {json.dumps({'type': 'class_imbalance_check', 'class_imbalance_check': class_imbalance_event})}\n\n"

        # After text stream, opportunistically generate a chart if the
        # message is about data and we have a dataset loaded
        if dataset_file_path:
            try:
                fp = Path(dataset_file_path)
                if fp.exists() and column_info:
                    df = pd.read_csv(fp)
                    chart = generate_chart_for_message(
                        body.message, df, column_info, full_response
                    )
                    if chart:
                        yield f"data: {json.dumps({'type': 'chart', 'chart': chart})}\n\n"
            except Exception:  # noqa: BLE001
                pass  # Charts are nice-to-have; never crash the chat

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/{project_id}/history")
def get_history(
    project_id: str,
    session: Session = Depends(get_session),
):
    statement = select(Conversation).where(Conversation.project_id == project_id)
    conversation = session.exec(statement).first()
    if not conversation:
        return {"messages": []}

    return {"messages": json.loads(conversation.messages)}


def _build_export_html(
    project: "Project",
    dataset: "Dataset | None",
    best_run: "ModelRun | None",
    messages: list[dict],
) -> str:
    """Generate a self-contained HTML analysis report from conversation history."""
    generated_at = datetime.now(UTC).strftime("%B %d, %Y at %H:%M UTC")
    dataset_info = ""
    if dataset:
        dataset_info = (
            f"<p><strong>Dataset:</strong> {dataset.filename} &nbsp;·&nbsp; "
            f"{dataset.row_count:,} rows, {dataset.column_count} columns</p>"
        )

    model_section = ""
    if best_run and best_run.status == "done":
        metrics = json.loads(best_run.metrics or "{}")
        primary_metric_key = next((k for k in ("r2", "accuracy") if k in metrics), None)
        metric_str = ""
        if primary_metric_key:
            val = metrics[primary_metric_key]
            label = "R²" if primary_metric_key == "r2" else "Accuracy"
            metric_str = f" &nbsp;·&nbsp; {label}: {val:.3f}"
        model_section = f"""
        <div class="model-box">
          <h2>Model Results</h2>
          <p><strong>Algorithm:</strong> {best_run.algorithm.replace("_", " ").title()}{metric_str}</p>
          {f'<p class="summary">{best_run.summary}</p>' if best_run.summary else ""}
        </div>"""

    # Render messages — skip empty content
    msg_html_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        ts = msg.get("timestamp", "")
        ts_str = (
            f'<span class="ts">{ts[:16].replace("T", " ") if ts else ""}</span>'
            if ts
            else ""
        )
        css_class = "user-msg" if role == "user" else "assistant-msg"
        label = "You" if role == "user" else "AutoModeler"
        # Escape HTML in message content
        safe_content = (
            content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        msg_html_parts.append(
            f'<div class="{css_class}"><div class="msg-header"><strong>{label}</strong>{ts_str}</div>'
            f'<div class="msg-body">{safe_content}</div></div>'
        )

    msgs_html = (
        "\n".join(msg_html_parts)
        if msg_html_parts
        else "<p><em>No messages in this conversation.</em></p>"
    )
    msg_count = len([m for m in messages if m.get("role") == "assistant"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AutoModeler Analysis: {project.name}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f9fafb; color: #111827; line-height: 1.6; padding: 2rem; }}
    .container {{ max-width: 800px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; font-weight: 700; color: #111827; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.1rem; font-weight: 600; color: #374151; margin-bottom: 0.75rem; }}
    .meta {{ color: #6b7280; font-size: 0.875rem; margin-bottom: 1.5rem; }}
    .meta p {{ margin-bottom: 0.25rem; }}
    .model-box {{ background: #f0fdf4; border: 1px solid #86efac; border-radius: 0.5rem;
                  padding: 1rem 1.25rem; margin-bottom: 1.5rem; }}
    .model-box .summary {{ color: #374151; margin-top: 0.5rem; font-size: 0.9rem; }}
    .section-label {{ font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
                      letter-spacing: 0.05em; color: #6b7280; margin-bottom: 0.75rem; }}
    .conversation {{ display: flex; flex-direction: column; gap: 0.75rem; }}
    .user-msg, .assistant-msg {{ border-radius: 0.5rem; padding: 0.875rem 1rem; }}
    .user-msg {{ background: #eff6ff; border: 1px solid #bfdbfe; margin-left: 2rem; }}
    .assistant-msg {{ background: #fff; border: 1px solid #e5e7eb; margin-right: 2rem; }}
    .msg-header {{ display: flex; justify-content: space-between; align-items: center;
                   margin-bottom: 0.35rem; }}
    .msg-header strong {{ font-size: 0.8rem; font-weight: 600; color: #374151; }}
    .ts {{ font-size: 0.7rem; color: #9ca3af; }}
    .msg-body {{ font-size: 0.9rem; color: #1f2937; }}
    .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
               text-align: center; font-size: 0.75rem; color: #9ca3af; }}
    @media print {{
      body {{ background: #fff; padding: 1rem; }}
      .user-msg {{ background: #f0f7ff; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Analysis Report: {project.name}</h1>
    <div class="meta">
      <p>Generated {generated_at}</p>
      {dataset_info}
      <p>{msg_count} AI response{"" if msg_count == 1 else "s"} in this conversation</p>
    </div>
    {model_section}
    <p class="section-label">Conversation Transcript</p>
    <div class="conversation">
      {msgs_html}
    </div>
    <div class="footer">Generated by <strong>AutoModeler</strong> — AI-powered data modeling for business analysts</div>
  </div>
</body>
</html>"""


@router.get("/{project_id}/export")
def export_conversation(
    project_id: str,
    session: Session = Depends(get_session),
):
    """Return conversation history as a downloadable self-contained HTML report."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Load conversation messages
    conv_stmt = select(Conversation).where(Conversation.project_id == project_id)
    conversation = session.exec(conv_stmt).first()
    messages: list[dict] = json.loads(conversation.messages) if conversation else []

    # Load dataset (most recent for this project)
    ds_stmt = select(Dataset).where(Dataset.project_id == project_id)
    dataset = session.exec(ds_stmt).first()

    # Load best completed model run
    runs_stmt = (
        select(ModelRun)
        .where(ModelRun.project_id == project_id)
        .where(ModelRun.status == "done")
    )
    runs = session.exec(runs_stmt).all()
    best_run: ModelRun | None = None
    if runs:
        # Prefer selected, then highest primary metric
        selected = [r for r in runs if r.is_selected]
        if selected:
            best_run = selected[0]
        else:

            def _primary(r: ModelRun) -> float:
                m = json.loads(r.metrics or "{}")
                return float(m.get("r2", m.get("accuracy", 0)))

            best_run = max(runs, key=_primary)

    html_content = _build_export_html(project, dataset, best_run, messages)
    safe_name = project.name.replace(" ", "_").replace("/", "-")[:40]
    filename = f"automodeler_{safe_name}_analysis.html"
    return HTMLResponse(
        content=html_content,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
