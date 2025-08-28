import uuid
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    id: int
    name: str
    X: Union[pd.DataFrame, np.ndarray]
    y: Union[pd.Series, np.ndarray]
    target_label: Optional[str] = None
