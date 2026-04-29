import torch.nn

from cd_mm_sits.lightning_module.losses import Sar2SarLoss, ShiftInvariantLoss


def test_shift_invariant_loss():
    l2 = torch.nn.MSELoss()
    loss = ShiftInvariantLoss(inner=l2, max_shift=1)

    input = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])
    target = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])

    result = loss(input, target)
    expected = 0.0
    assert torch.isclose(result, torch.tensor(expected)), (
        f"Expected {expected}, got {result}"
    )


def test_shift_invariant_loss_zeromaxshift():
    l2 = torch.nn.MSELoss()
    loss = ShiftInvariantLoss(inner=l2, max_shift=0)

    input = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])
    target = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])

    result = loss(input, target)
    expected = 1.0
    assert torch.isclose(result, torch.tensor(expected)), (
        f"Expected {expected}, got {result}"
    )


def test_shift_invariant_loss_different_input():
    l2 = torch.nn.MSELoss()
    loss = ShiftInvariantLoss(inner=l2, max_shift=1)

    input = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    target = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])

    result = loss(input, target)
    expected = 1.0
    assert torch.isclose(result, torch.tensor(expected)), (
        f"Expected {expected}, got {result}"
    )


def test_sar2sar_loss():
    loss1 = Sar2SarLoss(input_unit="amplitude")
    loss2 = Sar2SarLoss(input_unit="log-intensity")

    input = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]) + 0.2
    target = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])

    inputlog = torch.log(input**2)
    targetlog = torch.log(target**2)

    l1 = loss1(input, target)
    l2 = loss2(inputlog, targetlog)

    assert torch.isclose(l1, l2), f"Expected {l1}, got {l2}"
