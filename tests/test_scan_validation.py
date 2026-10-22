import numpy as np
import pytest
from scan_validation import validate_scan

@pytest.mark.parametrize("volume",[None,np.zeros((20,160,160)),np.zeros((5,160,160)),np.zeros((20,30,30)),np.full((20,160,160),np.nan),np.zeros((20,160,160),dtype="U1")])
def test_rejects_invalid(volume):
    assert validate_scan(volume)[0] is False
def test_accepts_shape_without_claiming_anatomy():
    volume=np.arange(20*160*160,dtype=np.float32).reshape(20,160,160)
    assert validate_scan(volume)[0] is True
