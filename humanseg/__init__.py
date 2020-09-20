from .unet import HumanSegNet
from .util import load_image, load_with_openpose, crop_image, get_mask_to_tensor, get_image_to_tensor
import torch
import cv2
import numpy as np


def infer(path,
          openpose_json=None,
          device='cuda:0',
          person_id=0,
          load_size=512):
    if openpose_json is not None:
        bbox, image = load_with_openpose(path,
                                         openpose_json,
                                         person_id=person_id,
                                         load_size=load_size)
    else:
        bbox, image = load_image(path, load_size=load_size)
    net = HumanSegNet().to(device=device)
    net.eval()
    with torch.no_grad():
        mask = net(image.to(device=device))
    return bbox, image, mask
