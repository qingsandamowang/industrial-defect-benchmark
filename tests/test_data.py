from pathlib import Path


def test_discover_pairs_matches_each_image_with_its_mask(tmp_path: Path):
    from defectbench.data import discover_pairs

    images = tmp_path / "MT_Crack" / "Imgs"
    images.mkdir(parents=True)
    (images / "sample.jpg").write_bytes(b"image")
    (images / "sample.png").write_bytes(b"mask")

    samples = discover_pairs(tmp_path)

    assert len(samples) == 1
    assert samples[0].category == "MT_Crack"
    assert samples[0].image.name == "sample.jpg"
    assert samples[0].mask.name == "sample.png"


def test_discover_pairs_rejects_missing_mask(tmp_path: Path):
    import pytest
    from defectbench.data import discover_pairs

    images = tmp_path / "MT_Crack" / "Imgs"
    images.mkdir(parents=True)
    (images / "sample.jpg").write_bytes(b"image")

    with pytest.raises(ValueError, match="sample"):
        discover_pairs(tmp_path)


def test_split_is_deterministic_disjoint_and_stratified(tmp_path: Path):
    from defectbench.data import Sample, split_samples

    samples = [
        Sample(category=category, image=tmp_path / f"{category}_{i}.jpg", mask=tmp_path / f"{category}_{i}.png")
        for category in ("MT_Crack", "MT_Free")
        for i in range(20)
    ]

    first = split_samples(samples, seed=42)
    second = split_samples(list(reversed(samples)), seed=42)

    assert first == second
    for category in ("MT_Crack", "MT_Free"):
        assert sum(s.category == category for s in first["train"]) == 14
        assert sum(s.category == category for s in first["val"]) == 3
        assert sum(s.category == category for s in first["test"]) == 3
    assert len({s.image for group in first.values() for s in group}) == 40
