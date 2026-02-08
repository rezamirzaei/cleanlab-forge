"""
Data Providers for Cleanlab Tasks.

This module provides concrete implementations of data providers for various datasets.
These can be customized or replaced to use different datasets with the same task logic.
"""

from cleanlab_demo.data.providers.multiannotator import (
    MovieLens100KProvider,
)
from cleanlab_demo.data.providers.multiclass import (
    CovtypeDataProvider,
    SKLearnDatasetProvider,
)
from cleanlab_demo.data.providers.multilabel import (
    EmotionsDataProvider,
    OpenMLMultilabelProvider,
)
from cleanlab_demo.data.providers.outlier import (
    CaliforniaHousingOutlierProvider,
)
from cleanlab_demo.data.providers.regression import (
    BikeSharingDataProvider,
    CaliforniaHousingDataProvider,
    TabularRegressionProvider,
)
from cleanlab_demo.data.providers.token import (
    ConlluDataProvider,
    UDEnglishEWTProvider,
)
from cleanlab_demo.data.providers.vision import (
    PennFudanPedProvider,
)

__all__ = [
    "BikeSharingDataProvider",
    "CaliforniaHousingDataProvider",
    "CaliforniaHousingOutlierProvider",
    "ConlluDataProvider",
    "CovtypeDataProvider",
    "EmotionsDataProvider",
    "MovieLens100KProvider",
    "OpenMLMultilabelProvider",
    "PennFudanPedProvider",
    "SKLearnDatasetProvider",
    "TabularRegressionProvider",
    "UDEnglishEWTProvider",
]
