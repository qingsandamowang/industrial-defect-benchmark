import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


def test_transfer_network_preserves_rectangular_geometry_and_has_gradients():
    from defectbench.transfer import ResNetUNet

    model = ResNetUNet(pretrained=False)
    logits = model(torch.rand(2, 1, 64, 96))
    assert logits.shape == (2, 1, 64, 96)
    logits.mean().backward()
    assert model.head.weight.grad is not None


def test_paired_augmentation_preserves_labels_and_does_not_mutate_source():
    from defectbench.experiment import paired_augment

    source = torch.zeros(1, 16, 16)
    source[:, 2:6, 3:8] = 1
    original = source.clone()
    for seed in range(8):
        torch.manual_seed(seed)
        image, mask = paired_augment(source, source, photometric=False)
        assert torch.equal(image, mask)
        assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(original, source)


def test_large_half_precision_dice_reduction_matches_float32():
    from defectbench.experiment import segmentation_loss

    logits = torch.zeros((1, 1, 384, 384))
    truth = torch.ones_like(logits)
    full = segmentation_loss(logits, truth)
    half = segmentation_loss(logits.half(), truth)
    assert torch.isfinite(half)
    assert half.item() == pytest.approx(full.item(), abs=2e-3)


def test_training_dataset_does_not_open_heldout_test_images(tmp_path, monkeypatch):
    from defectbench.data import discover_pairs, split_samples
    from defectbench.experiment import load_training_data
    import defectbench.training as training

    for category in ("MT_Free", "MT_Crack"):
        directory = tmp_path / category / "Imgs"
        directory.mkdir(parents=True)
        for index in range(6):
            for suffix in ("jpg", "png"):
                (directory / f"{index}.{suffix}").write_bytes(cv2.imencode("." + suffix, np.zeros((8, 12), np.uint8))[1].tobytes())
    opened = []
    real_read = training.read_grayscale
    monkeypatch.setattr(training, "read_grayscale", lambda path: (opened.append(path), real_read(path))[1])
    train, validation, groups = load_training_data(tmp_path, size=32, seed=42)
    assert len(train) + len(validation) < len(discover_pairs(tmp_path))
    test_paths = {sample.image for sample in split_samples(discover_pairs(tmp_path))["test"]}
    assert not test_paths.intersection(opened)
    assert len(groups["test"]) > 0


def test_dataset_fingerprint_detects_same_name_content_replacement(tmp_path):
    from defectbench.data import Sample
    from defectbench.experiment import content_manifest

    image, mask = tmp_path / "image.jpg", tmp_path / "image.png"
    image.write_bytes(b"image-v1")
    mask.write_bytes(b"mask-v1")
    groups = {"train": [], "val": [], "test": [Sample("MT_Crack", image, mask)]}
    first = content_manifest(groups)
    mask.write_bytes(b"mask-v2")
    second = content_manifest(groups)
    assert first["dataset_sha256"] != second["dataset_sha256"]
    assert first["samples"][0]["image_sha256"] == second["samples"][0]["image_sha256"]


def test_data_discovery_rejects_duplicate_keys_from_multiple_archive_copies(tmp_path):
    from defectbench.data import discover_pairs

    for copy in ("first", "second"):
        directory = tmp_path / copy / "MT_Crack/Imgs"
        directory.mkdir(parents=True)
        (directory / "sample.jpg").write_bytes(b"image")
        (directory / "sample.png").write_bytes(b"mask")
    with pytest.raises(ValueError, match="Duplicate"):
        discover_pairs(tmp_path)


def test_checkpoint_prediction_roundtrip_restores_unicode_source_geometry(tmp_path):
    from defectbench.model import UNet
    from defectbench.transfer import save_checkpoint, load_checkpoint
    from defectbench.predict import predict_inputs

    source = tmp_path / "输入" / "工件.jpg"
    source.parent.mkdir()
    source.write_bytes(cv2.imencode(".jpg", np.full((17, 31), 120, np.uint8))[1].tobytes())
    checkpoint = tmp_path / "model.pt"
    metadata = {"architecture": "unet", "base_channels": 4, "size": 32, "mask_threshold": 0.6, "image_threshold": 0.4}
    save_checkpoint(checkpoint, UNet(base_channels=4), metadata)
    model, loaded = load_checkpoint(checkpoint)
    assert loaded["mask_threshold"] == 0.6
    output = tmp_path / "output"
    result = predict_inputs(checkpoint, source, output)
    assert result["count"] == 1
    mask = cv2.imdecode(np.frombuffer((output / "工件_mask.png").read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (17, 31)
    assert set(np.unique(mask)) <= {0, 255}
    assert result["items"][0]["width"] == 31
    assert (output / "predictions.json").is_file()
    assert (output / "index.html").is_file()


def test_prediction_rejects_output_within_input_directory(tmp_path):
    from defectbench.predict import validate_output_location

    source = tmp_path / "images"
    source.mkdir()
    with pytest.raises(ValueError, match="output"):
        validate_output_location(source, source / "predictions")


def test_image_analysis_reports_actual_confusion_and_average_precision():
    from defectbench.analysis import detection_analysis

    report = detection_analysis([True, True, False, False], [0.9, 0.7, 0.8, 0.1], threshold=0.5)
    assert report["confusion"] == {"tp": 2, "fp": 1, "fn": 0, "tn": 1}
    assert report["false_positive_rate"] == 0.5
    assert report["average_precision"] == pytest.approx((1 + 2 / 3) / 2)


def test_html_report_escapes_titles_and_contains_no_external_dependency(tmp_path):
    from defectbench.html_report import write_report

    output = tmp_path / "report.html"
    write_report(output, {"count": 2, "pixel_micro_dice": 0.75, "per_category": {}}, title="<script>test</script>")
    text = output.read_text(encoding="utf-8")
    assert "&lt;script&gt;test&lt;/script&gt;" in text
    assert "0.750" in text
    assert '<script src=' not in text
