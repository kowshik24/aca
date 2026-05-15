import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def chunked(xs: list, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]
