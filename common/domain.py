import uuid
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Dataset:
    id: int
    name: str
    X: pd.DataFrame
    y: pd.DataFrame
    target_label: Optional[str] = None
