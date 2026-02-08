"""Cleanlab task implementations used by the UI and notebooks."""

from cleanlab_demo.tasks.base import DemoConfig, DemoResult
from cleanlab_demo.tasks.multiannotator import (
    MultiannotatorConfig,
    MultiannotatorResult,
    MultiannotatorTask,
)
from cleanlab_demo.tasks.multiclass import (
    MulticlassClassificationConfig,
    MulticlassClassificationResult,
    MulticlassClassificationTask,
)
from cleanlab_demo.tasks.multilabel import (
    MultilabelClassificationConfig,
    MultilabelClassificationResult,
    MultilabelClassificationTask,
)
from cleanlab_demo.tasks.outlier import (
    OutlierDetectionConfig,
    OutlierDetectionResult,
    OutlierDetectionTask,
)
from cleanlab_demo.tasks.regression import (
    RegressionCleanLearningConfig,
    RegressionCleanLearningResult,
    RegressionCleanLearningTask,
)
from cleanlab_demo.tasks.token import (
    TokenClassificationConfig,
    TokenClassificationResult,
    TokenClassificationTask,
)
from cleanlab_demo.tasks.vision import (
    VisionDetectionSegmentationConfig,
    VisionDetectionSegmentationResult,
    VisionDetectionSegmentationTask,
)

__all__ = [
    "DemoConfig",
    "DemoResult",
    "MultiannotatorConfig",
    "MultiannotatorResult",
    "MultiannotatorTask",
    "MulticlassClassificationConfig",
    "MulticlassClassificationResult",
    "MulticlassClassificationTask",
    "MultilabelClassificationConfig",
    "MultilabelClassificationResult",
    "MultilabelClassificationTask",
    "OutlierDetectionConfig",
    "OutlierDetectionResult",
    "OutlierDetectionTask",
    "RegressionCleanLearningConfig",
    "RegressionCleanLearningResult",
    "RegressionCleanLearningTask",
    "TokenClassificationConfig",
    "TokenClassificationResult",
    "TokenClassificationTask",
    "VisionDetectionSegmentationConfig",
    "VisionDetectionSegmentationResult",
    "VisionDetectionSegmentationTask",
]
