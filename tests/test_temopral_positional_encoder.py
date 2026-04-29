import torch

from cd_mm_sits.model.temporal_positional_encoder import TimeDeltaRotaryPE


def test_timedelta_rope():
    d = 64
    base = 1000
    b = 2

    rope = TimeDeltaRotaryPE(base, d)
    x = torch.ones(b, d)
    out = rope(x, delta=torch.ones(b) * 2)
    out2 = rope(x, delta=torch.ones(b) * 2)
    assert out.shape == (b, d)
    assert torch.allclose(out, out2, atol=1e-4, rtol=1e-4)
