"""SQLModel table models — imported here so all tables register before create_all."""

from models.project import Project
from models.dataset import Dataset
from models.feature_set import FeatureSet
from models.conversation import Conversation
from models.model_run import ModelRun
from models.deployment import Deployment
from models.prediction_log import PredictionLog
from models.feedback_record import FeedbackRecord

__all__ = [
    "Project",
    "Dataset",
    "FeatureSet",
    "Conversation",
    "ModelRun",
    "Deployment",
    "PredictionLog",
    "FeedbackRecord",
]
