def test_unet_output_matches_input_spatial_size_and_backpropagates():
    import torch
    from defectbench.model import UNet

    model = UNet(base_channels=4)
    x = torch.rand(2, 1, 64, 64)
    logits = model(x)
    logits.mean().backward()

    assert logits.shape == (2, 1, 64, 64)
    assert any(parameter.grad is not None for parameter in model.parameters())
