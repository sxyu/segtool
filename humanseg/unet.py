import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class HumanSegNet(nn.Module):
    '''
    A wrapper around smp.Unet
    '''
    def __init__(
            self,
            backbone='resnet34',
            model_url='https://ocf.berkeley.edu/~sxyu/humansegnet_latest.pt',
            latent_size=256):
        super().__init__()
        self.model = smp.Unet(
            backbone,
            encoder_weights='imagenet',
            classes=1,
        )

        state_dict = load_state_dict_from_url(model_url)
        self.model.load_state_dict(state_dict)
        self.latent_size = latent_size

    def encode(self, x):
        '''
        Get PIFu features
        '''
        features = self.model.encoder(x)
        # Scale and concat features
        scaled_features = []
        latent_shape = (self.latent_size, self.latent_size)
        for feat in features[1:]:
            scaled_features.append(
                F.interpolate(feat,
                              latent_shape,
                              align_corners=False,
                              mode='bilinear'))
        latent = torch.cat(scaled_features, dim=1)
        return latent

    def forward(self, x):
        '''
        Infer mask from image
        :param x (B, 3, H, W)
        :return mask (B, 1, H, W)
        '''
        masks = self.model(x)
        return torch.sigmoid(masks)
