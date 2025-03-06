import pytest
import numpy as np
from biotuner.biotuner_object import compute_biotuner
from biotuner.biotuner_utils import chunk_ts
from biotuner.metrics import compute_subharmonics_2lists
from biotuner.transitional_harmony import transitional_harmony

# Sample test data
@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.randn(2560)  # Simulated time series of 10 seconds

@pytest.fixture
def trans_harmony_instance(sample_data):
    return transitional_harmony(sf=256, data=sample_data, precision=0.5, n_harm=10, mode="win_overlap")

def test_initialization(trans_harmony_instance):
    assert trans_harmony_instance.sf == 256
    assert trans_harmony_instance.precision == 0.5
    assert trans_harmony_instance.n_harm == 10
    assert trans_harmony_instance.mode == "win_overlap"

def test_compute_trans_harmony_structure(trans_harmony_instance):
    trans_subharm, time_vec_final, subharm_melody = trans_harmony_instance.compute_trans_harmony()
    
    assert isinstance(trans_subharm, list)
    assert isinstance(time_vec_final, list)
    assert isinstance(subharm_melody, list)
    
    if len(trans_subharm) > 0:
        assert all(isinstance(x, float) or np.isnan(x) for x in trans_subharm)
        assert all(isinstance(t, float) for t in time_vec_final)
        assert all(isinstance(s, tuple) or np.isnan(s) for s in subharm_melody)

def test_mode_IF(trans_harmony_instance):
    trans_subharm, time_vec_final, subharm_melody = trans_harmony_instance.compute_trans_harmony(mode="IF")
    
    assert isinstance(trans_subharm, list)
    assert isinstance(time_vec_final, list)
    assert isinstance(subharm_melody, list)

def test_short_time_series():
    short_data = np.random.randn(50)  # Very short time series
    trans_harmony = transitional_harmony(sf=256, data=short_data)
    
    trans_subharm, time_vec_final, subharm_melody = trans_harmony.compute_trans_harmony()
    assert len(trans_subharm) == 0  # No windows should be formed

def test_long_time_series():
    long_data = np.random.randn(50000)  # Very long time series
    trans_harmony = transitional_harmony(sf=256, data=long_data)
    
    trans_subharm, time_vec_final, subharm_melody = trans_harmony.compute_trans_harmony()
    assert len(trans_subharm) > 0  # Should be able to process normally

def test_no_peaks_detected():
    zero_data = np.zeros(1000)  # All zero input
    trans_harmony = transitional_harmony(sf=256, data=zero_data)
    
    trans_subharm, time_vec_final, subharm_melody = trans_harmony.compute_trans_harmony()
    assert all(np.isnan(x) for x in trans_subharm)  # No valid subharmonics should be found

def test_compare_deltas(trans_harmony_instance):
    fig = trans_harmony_instance.compare_deltas([10, 20, 30])
    
    assert fig is not None  # Ensure the figure was created
