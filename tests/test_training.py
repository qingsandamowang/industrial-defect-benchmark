import cv2
import numpy as np


def test_tile_dataset_keeps_image_and_mask_aligned(tmp_path):
    from defectbench.data import Sample
    from defectbench.training import TileDataset

    image = np.full((10, 20), 180, dtype=np.uint8)
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 6:14] = 255
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "mask.png"
    image_path.write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
    mask_path.write_bytes(cv2.imencode(".png", mask)[1].tobytes())

    dataset = TileDataset([Sample("MT_Crack", image_path, mask_path)], size=32)
    x, y = dataset[0]

    assert tuple(x.shape) == (1, 32, 32)
    assert tuple(y.shape) == (1, 32, 32)
    assert y.max().item() == 1.0
    assert y.min().item() == 0.0
    assert dataset.has_defect == [True]


def test_combined_loss_has_finite_gradient():
    import torch
    from defectbench.training import combined_loss

    logits = torch.zeros((1, 1, 8, 8), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 2:4, 2:4] = 1
    loss = combined_loss(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_fit_model_saves_a_reloadable_checkpoint(tmp_path):
    import torch
    from defectbench.training import fit_model

    class TinyDataset:
        has_defect = [False, True, False, True]

        def __len__(self):
            return 4

        def __getitem__(self, index):
            x = torch.zeros((1, 32, 32))
            y = torch.zeros_like(x)
            if self.has_defect[index]:
                x[:, 12:20, 12:20] = 1
                y[:, 12:20, 12:20] = 1
            return x, y

    checkpoint = tmp_path / "model.pt"
    _, history = fit_model(TinyDataset(), TinyDataset(), checkpoint, epochs=1, batch_size=2, base_channels=4)

    assert checkpoint.is_file()
    assert len(history) == 1
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 1
    assert "model" in saved
