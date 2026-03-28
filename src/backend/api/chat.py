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
        m = re.search(rf"(\w[\w\s]{{0,30}}?)\s+{sep_pat}\s+(\w[\w\s]{{0,30}}?)(?:\s+(?:scatter|plot|chart|graph|$)|$)", msg_lower)
        if m:
            x_col = _match_col(m.group(1))
            y_col = _match_col(m.group(2))
            if x_col and y_col and x_col != y_col:
                return {"x_col": x_col, "y_col": y_col}

    # Pattern 2: "between X and Y"
    m = re.search(r"\bbetween\s+(\w[\w\s]{0,30}?)\s+and\s+(\w[\w\s]{0,30}?)(?:\s*[.?!]|$)", msg_lower)
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

# Extract "predict X" / "target is X" / "model for X" — scan against known columns
_TRAIN_TARGET_EXTRACT = re.compile(
    r"(?:"
    r"predict\s+(?:the\s+)?['\"]?(\w+)['\"]?|"
    r"target\s+(?:is\s+|column\s+(?:is\s+)?)?['\"]?(\w+)['\"]?|"
    r"(?:model|forecast|estimate)\s+(?:the\s+)?['\"]?(\w+)['\"]?"
    r")",
    re.IGNORECASE,
)


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
    r"set\s+(?:a\s+)?filter"
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

        # Emit deployed event — model is now live
        if deployed_event:
            yield f"data: {json.dumps({'type': 'deployed', 'deployment': deployed_event})}\n\n"

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

        # Emit record table viewer result
        if records_event:
            yield f"data: {json.dumps({'type': 'records', 'records': records_event})}\n\n"

        # Emit what-if prediction result
        if whatif_chat_event:
            yield f"data: {json.dumps({'type': 'whatif_result', 'whatif': whatif_chat_event})}\n\n"

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
