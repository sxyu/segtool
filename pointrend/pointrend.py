import os
import sys
import json
import tqdm

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
POINTREND_ROOT_PATH = os.path.join(ROOT_PATH, 'detectron2', 'projects',
                                   'PointRend')
sys.path.insert(0, POINTREND_ROOT_PATH)

# import PointRend project
import point_rend

import numpy as np
import cv2
#  from matplotlib import pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


class PointRendWrapper:
    def __init__(self, filter_class=-1):
        '''
        :param filter_class output only intances of filter_class (-1 to disable). Note: class 0 is person.
        '''
        self.filter_class = filter_class
        self.coco_metadata = MetadataCatalog.get("coco_2017_val")
        self.cfg = get_cfg()

        # Add PointRend-specific config

        point_rend.add_pointrend_config(self.cfg)

        # Load a config from file
        self.cfg.merge_from_file(
            os.path.join(
                POINTREND_ROOT_PATH,
                "configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
            ))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
        self.cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
        self.predictor = DefaultPredictor(self.cfg)

    def segment(self, im, out_name='', visualize=False):
        '''
        Run PointRend
        :param out_name if set, writes segments B&W mask to this image file
        :param visualize if set, and out_name is set, outputs visualization rater than B&W mask
        '''
        outputs = self.predictor(im)

        insts = outputs["instances"]
        if self.filter_class != -1:
            insts = insts[insts.pred_classes ==
                          self.filter_class]  # 0 is person
        if visualize:
            v = Visualizer(im[:, :, ::-1],
                           self.coco_metadata,
                           scale=1.2,
                           instance_mode=ColorMode.IMAGE_BW)

            point_rend_result = v.draw_instance_predictions(
                insts.to("cpu")).get_image()
            if out_name:
                cv2.imwrite(out_name + '.png', point_rend_result[:, :, ::-1])
            return point_rend_result[:, :, ::-1]
        else:
            im_names = []
            masks = []
            for i in range(len(insts)):
                mask = insts[i].pred_masks.to("cpu").permute(1, 2,
                                                             0).numpy() \
                        * np.uint8(255)
                if out_name:
                    im_name = out_name
                    if i:
                        im_name += '_' + str(i) + '.png'
                    else:
                        im_name += '.png'
                    im_names.append(im_name)
                    cv2.imwrite(im_name, mask)
                masks.append(mask)
            if out_name:
                with open(out_name + '.json', 'w') as fp:
                    json.dump({'files': im_names}, fp)
            return masks


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide an example image")
        sys.exit(1)
    COCO_CLASS_HUMAN = 0
    pointrend = PointRendWrapper(COCO_CLASS_HUMAN)
    for image_path in tqdm.tqdm(sys.argv[1:]):
        im = cv2.imread(image_path)
        masks = pointrend.segment(im)
        if len(masks) == 0:
            print("ERROR: PointRend detected no humans in", image_path,
                  "please try another image")
            sys.exit(1)
        mask_main = masks[0]
        assert mask_main.shape[:2] == im.shape[:2]
        assert mask_main.shape[-1] == 1
        assert mask_main.dtype == 'uint8'

        img_no_ext = os.path.splitext(image_path)[0]
        out_mask_path = img_no_ext + "_mask.png"
        cv2.imwrite(out_mask_path, mask_main)
