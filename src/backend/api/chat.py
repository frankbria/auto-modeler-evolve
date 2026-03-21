import json
import re
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from chat.orchestrator import build_system_prompt, detect_state, generate_suggestions
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
            .where(FeatureSet.dataset_id == dataset.id, FeatureSet.is_active == True)  # noqa: E712
            .order_by(FeatureSet.created_at.desc())  # type: ignore[arg-type]
        ).first()

    model_runs = list(
        session.exec(select(ModelRun).where(ModelRun.project_id == project_id)).all()
    )

    # Latest active deployment
    deployment = session.exec(
        select(Deployment)
        .where(Deployment.project_id == project_id, Deployment.is_active == True)  # noqa: E712
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
                    "metrics": json.loads(target_run.metrics)
                    if target_run.metrics
                    else {},
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
                _df = pd.read_csv(_file_path)
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
                _df = pd.read_csv(_file_path)
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
                _df = pd.read_csv(_file_path)
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
                _df = pd.read_csv(_file_path)
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
                _df = pd.read_csv(_file_path)
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
                _df = pd.read_csv(_file_path)
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
                _df = pd.read_csv(_file_path)
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
                _df = pd.read_csv(_file_path)
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
                _df = pd.read_csv(_file_path)
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
                    _df = pd.read_csv(_file_path)
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

    # Check for column rename request ("rename revenue_usd to Revenue")
    rename_result: dict | None = None
    if _RENAME_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = pd.read_csv(_file_path)
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

        # Emit follow-up suggestion chips (always, if we have any)
        if suggestions_list:
            yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions_list})}\n\n"

        # Emit correlation heatmap if triggered (reuses existing {type:"chart"} path)
        if heatmap_chart:
            yield f"data: {json.dumps({'type': 'chart', 'chart': heatmap_chart})}\n\n"

        # Emit column rename result if executed
        if rename_result:
            yield f"data: {json.dumps({'type': 'rename_result', 'rename': rename_result})}\n\n"

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
