import torch

from cd_mm_sits.layers.xpos import (
    XPOS,
    MyXpos,
    RetNetTimeXpos,
    RetNetXpos,
    TimeXPOS,
    _get_xi_xpos,
    _get_xi_xpostime,
    fixed_pos_embedding_3d,
)


def test_xpos_forward():
    x = torch.eye(4).unsqueeze(0)
    print(x.shape)
    xpos = XPOS(4)
    x_rot = xpos(x)

    # apply reverse
    x_rot_rev = xpos.forward(x)
    print(x_rot @ x_rot_rev.transpose(-1, -2))
    assert x_rot.shape == x.shape


def test_fixed_pos_embedding_3d():
    B, T, D = 2, 11, 10
    theta = torch.randn(D)
    output = fixed_pos_embedding_3d(theta, (B, T))
    assert output.shape == (B, T, D)
    assert output[0, 2, 0] == theta[0] * 2
    assert output[0, 3, 2] == theta[2] * 3


def test_get_xi_xpos():
    head_dim = 10
    gamma = 0.4
    scale = (torch.arange(0, head_dim, 2) + gamma * head_dim) / ((1 + gamma) * head_dim)
    T = 11
    scale_base = 512
    decay = _get_xi_xpos(scale, T, scale_base, scale.dtype, scale.device)
    assert decay.shape == (1, T, head_dim // 2), f"day {decay.shape}"


def test_get_xi_xpos_time():
    T = 11
    scale_base = 512
    B = 2
    decay = _get_xi_xpostime(torch.Tensor([0]), torch.randn(B, T), scale_base)
    assert decay.shape == (B, T, 1), f"day {decay.shape}"
    assert torch.all(~torch.isneginf(decay))


def test_my_xpos():
    scale_base = 512
    head_dim = 32
    my_xpos = MyXpos(scale_base=scale_base, embed_dim=head_dim)
    xpos = XPOS(head_dim, scale_base)
    B, T, D = 2, 11, head_dim
    x = torch.randn(B, T, D)
    out_xpos = xpos(x)
    out_my_xpos = my_xpos(x)
    dis_mse = torch.nn.MSELoss()
    assert dis_mse(out_my_xpos, out_xpos) < 1e-4
    # assert torch.allclose(
    #     out_xpos, out_my_xpos, atol=1e-3, rtol=1e-3
    # ), f"dis {dis_mse(out_xpos, out_my_xpos)}"


def test_my_time_xpos():
    scale_base = 512
    head_dim = 32
    my_xpos = TimeXPOS(scale_base=scale_base, embed_dim=head_dim)
    B, T, D = 2, 11, head_dim
    x = torch.ones(B, T, D)
    time = torch.arange(T).unsqueeze(0).expand(B, -1)
    print(x.shape, time.shape)
    out_my_xpos = my_xpos(x, time)
    assert out_my_xpos.shape == (B, T, D)
    assert torch.allclose(
        out_my_xpos[0, ...], out_my_xpos[1, ...], atol=1e-4, rtol=1e-4
    )
    assert torch.all(~torch.isneginf(out_my_xpos))


def test_my_time_retnetxpos():
    scale_base = 512
    head_dim = 32
    my_xpos = RetNetTimeXpos(scale_base=scale_base, h_index=0, embed_dim=head_dim)
    B, T, D = 2, 11, head_dim
    x = torch.ones(B, T, D)
    time = torch.arange(T).unsqueeze(0).expand(B, -1)
    print(x.shape, time.shape)
    out_my_xpos = my_xpos(x, time)
    assert out_my_xpos.shape == (B, T, D)
    assert torch.allclose(
        out_my_xpos[0, ...], out_my_xpos[1, ...], atol=1e-4, rtol=1e-4
    )


def test_my_retnetxpos():
    scale_base = 512
    head_dim = 32
    my_xpos = RetNetXpos(scale_base=scale_base, h_index=0, embed_dim=head_dim)
    B, T, D = 2, 11, head_dim
    x = torch.ones(B, T, D)
    time = torch.arange(T).unsqueeze(0).expand(B, -1)
    print(x.shape, time.shape)
    out_my_xpos = my_xpos(x)
    assert out_my_xpos.shape == (B, T, D)
    assert torch.allclose(
        out_my_xpos[0, ...], out_my_xpos[1, ...], atol=1e-4, rtol=1e-4
    )


# def test_time_xpos_forward():
#     B, T, C = 2, 11, 32
#     x = torch.randn(B, T, C)
#     # x = torch.eye(4).unsqueeze(0)
#     # print(x.shape)
#     time = torch.randn(B, T)
#     xpos = TimeXPOS(4)
#     x_rot = xpos(x, time=time)
#     # apply reverse
#     x_rot_rev = xpos.forward(x, time=torch.arange(4))

#     print(x_rot @ x_rot_rev.transpose(-1, -2))
#     assert x_rot.shape == x.shape
