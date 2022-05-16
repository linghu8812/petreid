import logging
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class SwinTransformer(nn.Module):
    def __init__(self, depth, pretrained=False):
        super().__init__()
        self.model = timm.create_model(depth, pretrained=pretrained)
        self.num_features = 768

    def forward(self, x):
        x = self.model.patch_embed(x)
        if self.model.absolute_pos_embed is not None:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        x = self.model.norm(x)  # B L C
        x = x.transpose(1, 2)  # B C 1
        # x = self.model.forward_features(x).unsqueeze(2).unsqueeze(2)
        return x


def swin_tiny_patch4_window7_224(pretrained=False):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    model = SwinTransformer('swin_tiny_patch4_window7_224', pretrained=pretrained)
    model.num_features = 768
    return model


def swin_base_patch4_window7_224(pretrained=False):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    model = SwinTransformer('swin_base_patch4_window7_224', pretrained=pretrained)
    model.num_features = 1024
    return model


def swin_large_patch4_window7_224_in22k(pretrained=False):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    model = SwinTransformer('swin_large_patch4_window7_224_in22k', pretrained=pretrained)
    model.num_features = 1536
    return model


def swin_base_patch4_window12_384_in22k(pretrained=False):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    model = SwinTransformer('swin_base_patch4_window12_384_in22k', pretrained=pretrained)
    model.num_features = 1024
    return model
