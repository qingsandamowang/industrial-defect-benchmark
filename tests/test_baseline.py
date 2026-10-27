import numpy as np


def test_dark_otsu_detects_large_dark_region():
    from defectbench.baseline import otsu_mask

    image = np.full((64, 64), 220, dtype=np.uint8)
    image[20:40, 20:40] = 10

    mask = otsu_mask(image, polarity="dark", kernel=1, min_area=10)

    assert mask[30, 30]
    assert not mask[0, 0]


def test_flat_image_has_no_defect_mask():
    from defectbench.baseline import otsu_mask

    assert not otsu_mask(np.full((32, 32), 128, dtype=np.uint8), polarity="dark").any()
