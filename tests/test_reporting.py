import numpy as np


def test_summarize_predictions_reports_micro_overlap_and_image_f1():
    from defectbench.reporting import summarize_predictions

    truths = [np.array([[1, 0], [0, 0]], dtype=bool), np.zeros((2, 2), dtype=bool)]
    predictions = [np.array([[1, 1], [0, 0]], dtype=bool), np.zeros((2, 2), dtype=bool)]
    scores = [0.9, 0.1]
    categories = ["MT_Crack", "MT_Free"]

    report = summarize_predictions(truths, predictions, scores, categories, image_threshold=0.5)

    assert report["pixel_micro_iou"] == 0.5
    assert report["pixel_micro_dice"] == 2 / 3
    assert report["image_f1"] == 1.0
    assert report["per_category"]["MT_Crack"]["count"] == 1
