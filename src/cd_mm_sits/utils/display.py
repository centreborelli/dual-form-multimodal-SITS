from torchvision.transforms.functional import normalize


def unscale_data(stats, sits):
    # sits dim n c h w
    n, c, h, w = sits.shape
    median = stats.median
    scale = [qmax - qmin for qmax, qmin in zip(stats.qmax, stats.qmin, strict=False)]
    sits = normalize(sits, mean=[0] * c, std=[1 / elem for elem in scale])
    sits = normalize(sits, mean=[-elem for elem in median], std=[1] * c)
    return sits
