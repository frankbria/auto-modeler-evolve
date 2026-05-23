"""Deployment changelog — immutable audit log of changes made to a deployment.

Each entry records a single change event (deploy, retrain, undeploy, api_key_added, …)
with a human-readable description.  Entries are written in the API layer and are
never modified or deleted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional

from sqlmodel import Field, SQLModel

# ── change_type constants ────────────────────────────────────────────────────
# Keep as plain strings so they survive across DB reads without an Enum import.
CHANGE_DEPLOYED = "deployed"
CHANGE_REDEPLOYED = "redeployed"
CHANGE_RETRAINED = "retrained"
CHANGE_UNDEPLOYED = "undeployed"
CHANGE_API_KEY_ADDED = "api_key_added"
CHANGE_API_KEY_REMOVED = "api_key_removed"
CHANGE_RATE_LIMIT_SET = "rate_limit_set"
CHANGE_QUOTA_SET = "quota_set"
CHANGE_ALERT_RULE_ADDED = "alert_rule_added"
CHANGE_FIELD_CONFIG_UPDATED = "field_config_updated"
CHANGE_BATCH_RUN_COMPLETE = "batch_run_complete"

ALL_CHANGE_TYPES = {
    CHANGE_DEPLOYED,
    CHANGE_REDEPLOYED,
    CHANGE_RETRAINED,
    CHANGE_UNDEPLOYED,
    CHANGE_API_KEY_ADDED,
    CHANGE_API_KEY_REMOVED,
    CHANGE_RATE_LIMIT_SET,
    CHANGE_QUOTA_SET,
    CHANGE_ALERT_RULE_ADDED,
    CHANGE_FIELD_CONFIG_UPDATED,
    CHANGE_BATCH_RUN_COMPLETE,
}

# Maximum entries returned by the changelog endpoint
CHANGELOG_MAX_ENTRIES = 50


class DeploymentChangelog(SQLModel, table=True):
    """Immutable audit log entry for a deployment."""

    __tablename__ = "deploymentchangelog"

    id: Optional[str] = Field(default=None, primary_key=True)
    deployment_id: str = Field(index=True)
    change_type: str  # one of the CHANGE_* constants
    description: str  # plain-English explanation of what changed
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
