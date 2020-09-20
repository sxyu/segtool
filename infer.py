import sys
import humanseg
import torch
from matplotlib import pyplot as plt
import numpy as np

assert len(sys.argv) > 1

bbox, image, mask = humanseg.infer(sys.argv[1],
                                   sys.argv[2] if len(sys.argv) > 2 else None)

mask = mask.cpu().numpy()[0, 0, :, :, None]
image_vis = (image[0].permute(1, 2, 0) + 1) * 0.5

color = np.array((1.0, 0.0, 0.0))
alpha = 0.5

colored_mask = image_vis * mask * (1.0 - alpha) + mask * color * alpha
image_vis = image_vis * (1.0 - mask) + colored_mask

plt.imshow(image_vis)
plt.show()
