import logging
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)


class EfficientNetModel(nn.Module):
    def __init__(self, depth, pretrained=False):
        super().__init__()
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(depth)
        else:
            self.backbone = EfficientNet.from_name(depth)
        self.backbone._blocks[38]._depthwise_conv.stride = [1, 1]
        self.backbone._blocks[38]._depthwise_conv.static_padding.padding = (2, 2, 2, 2)
        self.num_features = 768

    def forward(self, x):
        return self.backbone.extract_features(x)


def efficientnetb4(pretrained=False):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    model = EfficientNetModel('efficientnet-b4', pretrained=pretrained)
    model.num_features = 1792
    return model


def efficientnetb7(pretrained=False):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    model = EfficientNetModel('efficientnet-b7', pretrained=pretrained)
    model.num_features = 2560
    return model
