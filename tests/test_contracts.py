"""Collect the offline contract suite (the same one the notebook runs in section 9A)."""
from tago.workflows.contracts import OfflineContractTests, V38RegressionTests  # noqa: F401  (pytest collects unittest classes)
