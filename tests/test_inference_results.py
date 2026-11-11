"""Contract tests for the committed inference results the dashboard reads (rule R19: CI on every push).

The README's Phase 3 numbers must be reproducible from data/inference_results.parquet.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

DATA = Path(__file__).resolve().parents[1] / "data" / "inference_results.parquet"


@pytest.fixture(scope="module")
def d():
    return pd.read_parquet(DATA)


def test_schema_and_splits(d):
    need = {"tile_id", "event", "split", "true_label", "pred_multimodal", "pred_optical"}
    assert need <= set(d.columns) and d.tile_id.is_unique
    assert set(d.split) == {"train", "val", "test"} and set(d.true_label) <= {0, 1, 2}


@pytest.mark.parametrize("model", ["mm", "opt"])
def test_confidences_are_probabilities(d, model):
    c = d[[f"conf_{model}_intact", f"conf_{model}_damaged", f"conf_{model}_destroyed"]]
    assert ((c >= 0) & (c <= 1)).all().all() and np.allclose(c.sum(axis=1), 1, atol=1e-3)


@pytest.mark.parametrize("split,mm,opt", [("val", 0.580, 0.558), ("test", 0.454, 0.512)])
def test_readme_macro_f1_is_reproducible(d, split, mm, opt):
    s = d[d.split == split]
    f = lambda p: f1_score(s.true_label, s[p], average="macro", labels=[0, 1, 2])
    assert round(f("pred_multimodal"), 3) == mm and round(f("pred_optical"), 3) == opt
