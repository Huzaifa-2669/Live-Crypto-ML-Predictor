import pandas as pd
import pytest
from src.utils import quality_check


def test_quality_check_passes_minimal_df():
    df = pd.DataFrame(
        {
            "date": [0, 1],
            "rate": [1.0, 2.0],
            "volume": [10, 20],
            "cap": [100, 200],
        }
    )
    quality_check(df)  # should not raise


def test_quality_check_missing_col_raises():
    df = pd.DataFrame({"date": [0], "rate": [1.0]})
    with pytest.raises(ValueError):
        quality_check(df)
