import numpy as np
import pytest
from src.data import load_plant
from src.twin import fit_twin, predict

@pytest.fixture(scope="module")
def plant2():
    return load_plant(2)[0]

@pytest.fixture(scope="module")
def twin2(plant2):
    return fit_twin(plant2)

def test_load_plant_schema_and_time(plant2):
    for c in ["DATE_TIME", "INVERTER", "SOURCE_KEY", "DC_POWER", "AC_POWER"]:
        assert c in plant2.columns
    assert plant2.DATE_TIME.is_monotonic_increasing

def test_dc_rescaling():
    df, _ = load_plant(1)
    r = (df.DC_POWER / df.AC_POWER.replace(0, np.nan)).dropna()
    assert r.median() < 2.0

def test_twin_has_per_inverter_models(plant2, twin2):
    assert len(twin2["models"]) == plant2.INVERTER.nunique()
    pred = predict(twin2, plant2.head(100))
    assert np.isfinite(pred).all()

def test_zero_irradiance_is_zero(plant2, twin2):
    night = plant2[plant2.IRRADIATION <= 0].head(20)
    if len(night):
        assert np.allclose(predict(twin2, night), 0.0)
