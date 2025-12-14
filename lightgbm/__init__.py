from __future__ import annotations

from pathlib import Path
import importlib
import sys

_project_root = Path(__file__).resolve().parents[1]
_src_package = _project_root / "src" / "lightgbm"
if _src_package.exists():
    __path__.insert(0, str(_src_package))

try:
    from .lgbm_regressor import LGBMRegressor  # type: ignore
    from .lgbm_classifier import LGBMClassifier  # type: ignore
except Exception:
    LGBMRegressor = None  # type: ignore
    LGBMClassifier = None  # type: ignore
try:
    _loss_mod = importlib.import_module("lightgbm.loss_functions")
    if not hasattr(_loss_mod, "HUBERLoss") and hasattr(_loss_mod, "HuberLoss"):
        setattr(_loss_mod, "HUBERLoss", getattr(_loss_mod, "HuberLoss"))
    if not hasattr(_loss_mod, "QUANTILELoss") and hasattr(_loss_mod, "QuantileLoss"):
        setattr(_loss_mod, "QUANTILELoss", getattr(_loss_mod, "QuantileLoss"))
except Exception:
    pass

__all__ = ["LGBMRegressor", "LGBMClassifier"]
