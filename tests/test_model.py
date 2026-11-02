"""UNet 结构的单元测试（CPU 即可运行，无需数据）。

覆盖：任意尺寸输入（含非 16 倍数）、批量推理、输出值域、参数规模。
"""
import torch

from unet import UNet


def test_non_multiple_of_16_input() -> None:
    """输入边长不是 16 的倍数时不应崩溃，且输出尺寸与输入一致。"""
    model = UNet().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 100, 130))
    assert out.shape == (1, 1, 100, 130)


def test_square_input_shape() -> None:
    model = UNet().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_output_is_logits() -> None:
    """网络输出必须是未过 sigmoid 的 logits：结构中不应存在 Sigmoid 层。"""
    model = UNet()
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())
    assert isinstance(model.outc, torch.nn.Conv2d)


def test_parameter_count() -> None:
    model = UNet()
    assert 30e6 < model.num_parameters < 32e6


def test_backward_runs() -> None:
    model = UNet()
    x = torch.randn(1, 3, 32, 32)
    loss = model(x).mean()
    loss.backward()
    assert model.inc.double_conv[0].weight.grad is not None
