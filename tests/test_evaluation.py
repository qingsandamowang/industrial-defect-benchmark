import numpy as np


def test_mask_threshold_is_chosen_from_validation_overlap():
    from defectbench.evaluation import choose_mask_threshold

    truths = [np.array([[1, 0]], dtype=bool)]
    probabilities = [np.array([[0.8, 0.4]], dtype=np.float32)]

    threshold = choose_mask_threshold(truths, probabilities, candidates=[0.3, 0.5, 0.9])

    assert threshold == 0.5


def test_image_threshold_is_chosen_by_validation_f1():
    from defectbench.evaluation import choose_image_threshold

    threshold = choose_image_threshold([False, True], [0.2, 0.8], candidates=[0.1, 0.5, 0.9])

    assert threshold == 0.5


def test_unet_inference_restores_original_shape(tmp_path):
    import cv2
    import torch
    from defectbench.data import Sample
    from defectbench.evaluation import infer_probabilities

    class ConstantModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    image = np.full((17, 30), 128, dtype=np.uint8)
    image_path = tmp_path / "sample.jpg"
    mask_path = tmp_path / "sample.png"
    image_path.write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
    mask_path.write_bytes(cv2.imencode(".png", image)[1].tobytes())

    maps, latencies = infer_probabilities(ConstantModel(), [Sample("MT_Free", image_path, mask_path)], size=32)

    assert maps[0].shape == image.shape
    assert np.allclose(maps[0], 0.5)
    assert latencies[0] >= 0
