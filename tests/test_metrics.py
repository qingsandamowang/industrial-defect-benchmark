import numpy as np


def test_mask_metrics_counts_pixels_and_computes_overlap():
    from defectbench.metrics import mask_metrics

    truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    result = mask_metrics(truth, prediction)

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["iou"] == 1 / 3
    assert result["dice"] == 1 / 2


def test_empty_truth_and_prediction_are_perfect_overlap():
    from defectbench.metrics import mask_metrics

    result = mask_metrics(np.zeros((2, 2)), np.zeros((2, 2)))

    assert result["iou"] == 1.0
    assert result["dice"] == 1.0


def test_image_metrics_uses_scores_and_threshold():
    from defectbench.metrics import image_metrics

    result = image_metrics([False, False, True, True], [0.1, 0.8, 0.7, 0.9], threshold=0.5)

    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0
    assert result["f1"] == 0.8
    assert result["auroc"] == 0.75
