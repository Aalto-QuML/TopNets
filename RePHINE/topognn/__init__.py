import os.path
from enum import Enum, auto
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_2d')


class Tasks(Enum):
    """Valid tasks."""

    GRAPH_CLASSIFICATION = auto()
    NODE_CLASSIFICATION = auto()
    NODE_CLASSIFICATION_WEIGHTED = auto()