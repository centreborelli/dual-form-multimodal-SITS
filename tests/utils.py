import torch

from cd_mm_sits.data.batch_class import CDSupBatch, DeforBatch


def rdn_batch(t1: int, t2: int, c: int, h: int, w: int) -> CDSupBatch:
    batch = torch.nested.nested_tensor(
        [torch.randn(t1, c, h, w), torch.randn(t2, c, h, w)],
        layout=torch.jagged,
    )
    doy = torch.nested.nested_tensor(
        [torch.arange(t1), torch.arange(t2)],
        layout=torch.jagged,
    )
    label = torch.nested.nested_tensor(
        [(torch.randn(t1, h, w) > 0.5).int(), (torch.randn(t2, h, w) > 0.5).int()],
        layout=torch.jagged,
    )
    return CDSupBatch(sits=batch, time=doy, label=label)


def rdn_defor_batch(t1: int, t2: int, c: int, h: int, w: int) -> DeforBatch:
    batch = torch.nested.nested_tensor(
        [
            torch.randn(t1, c, h, w, requires_grad=True),
            torch.randn(t2, c, h, w, requires_grad=True),
        ],
        layout=torch.jagged,
    )
    doy = torch.nested.nested_tensor(
        [torch.arange(t1), torch.arange(t2)],
        layout=torch.jagged,
    )
    label = torch.stack(
        [(torch.randn(h, w) > 0.5).int(), (torch.randn(h, w) > 0.5).int()]
    )
    return DeforBatch(
        sits=batch,
        time=doy,
        label=label,
        idx_after=[t1 - 1, t2 - 1],
        idx_bef=[1, 1],
        batch_idx=[0, 1],
    )
