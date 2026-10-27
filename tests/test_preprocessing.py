import numpy as np


def test_letterbox_and_restore_keep_original_geometry():
    from defectbench.preprocessing import letterbox, restore_map

    original = np.zeros((4, 8), dtype=np.uint8)
    original[1:3, 3:5] = 255

    prepared, geometry = letterbox(original, size=16, mask=True)
    restored = restore_map(prepared, geometry, mask=True)

    assert prepared.shape == (16, 16)
    assert restored.shape == original.shape
    assert np.array_equal(restored, original)


def test_letterbox_rejects_empty_images():
    import pytest
    from defectbench.preprocessing import letterbox

    with pytest.raises(ValueError):
        letterbox(np.zeros((0, 8), dtype=np.uint8), size=16)
