from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .generator import *
from .discriminator import *
from .spgan import *
from .swin_transformer import swin_tiny_patch4_window7_224, swin_base_patch4_window7_224, \
    swin_large_patch4_window7_224_in22k, swin_base_patch4_window12_384_in22k
from .efficientnet import efficientnetb4, efficientnetb7

__all__ = ["build_bakcbone", "names"]

__factory = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnet50_ibn_a": resnet50_ibn_a,
    "resnet101_ibn_a": resnet101_ibn_a,
    "resnet_6blocks": resnet_6blocks,
    "resnet_9blocks": resnet_9blocks,
    "patchgan_3layers": patchgan_3layers,
    "metricnet": metricnet,
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
    "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
    "swin_large_patch4_window7_224_in22k": swin_large_patch4_window7_224_in22k,
    "swin_base_patch4_window12_384_in22k": swin_base_patch4_window12_384_in22k,
    "efficientnetb4": efficientnetb4,
    "efficientnetb7": efficientnetb7,
}


def names():
    return sorted(__factory.keys())


def build_bakcbone(name, pretrained=True, *args, **kwargs):
    """
    Create a backbone model.
    Parameters
    ----------
    name : str
        The backbone name.
    pretrained : str
        ImageNet pretrained.
    """
    if name not in __factory:
        raise KeyError("Unknown backbone:", name)
    return __factory[name](pretrained=pretrained, *args, **kwargs)
