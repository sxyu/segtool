from torchvision import transforms
import cv2
import numpy as np
import json


def _load_openpose(openpose_file,
                   rescale=1.3,
                   pad=0.0,
                   detection_thresh=0.2,
                   min_keypoints=10,
                   person_id=0,
                   max_aspect=None):
    '''
      Load OpenPose json file
      :return keypoints, bbox
      '''
    with open(openpose_file, 'r') as f:
        people = json.load(f)['people']
        if len(people) == 0:
            print('Warning:', openpose_file, 'provides no detections')
            return None, None
        keypoints = people[person_id]['pose_keypoints_2d']
        keypoints = np.reshape(np.array(keypoints), (-1, 3))
    valid = keypoints[:, -1] > detection_thresh
    num_valid = valid.nonzero()[0].shape[0]
    if num_valid < min_keypoints:
        return None, None

    #  required_op_joints_ind = np.array([9, 12, 2, 5])
    #  if not valid[required_op_joints_ind].all():
    #      # A required joint is not available
    #      return None, None

    valid_keypoints = keypoints[valid][:, :-1]

    tl = valid_keypoints.min(axis=0)
    br = valid_keypoints.max(axis=0)
    center = np.round(valid_keypoints.mean(axis=0)).astype(int)
    sz = (br - tl) * rescale + pad

    if max_aspect is None:
        # Make it square
        sz[:] = sz.max()
    else:
        # Do not allow long rectangles
        if sz[1] / sz[0] > max_aspect:
            sz[0] = sz[1] / max_aspect
        elif sz[0] / sz[1] > max_aspect:
            sz[1] = sz[0] / max_aspect

    half_sz = np.ceil(sz * 0.5).astype(int)
    tl_trans = center - half_sz
    return keypoints, np.array(
        [tl_trans[0], tl_trans[1], half_sz[0] * 2, half_sz[1] * 2])


def crop_image(img, rect, border_shade=0):
    '''
    Image cropping helper
    '''
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1] - (x + w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0] - (y + h)) if y + h >= img.shape[0] else 0

    color = [border_shade] * img.shape[2]
    new_img = cv2.copyMakeBorder(img,
                                 top,
                                 bottom,
                                 left,
                                 right,
                                 cv2.BORDER_CONSTANT,
                                 value=color)
    if len(new_img.shape) == 2:
        new_img = new_img[..., None]

    x = x + left
    y = y + top

    return new_img[y:(y + h), x:(x + w), :]


def get_image_to_tensor():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.0, ), (1.0, ))])


def load_with_openpose(image, openpose_json_path, person_id=0, load_size=512):
    _, op_bbox = _load_openpose(openpose_json_path, person_id=person_id)

    if isinstance(image, str):
        image_bgr = cv2.imread(image)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_crop = crop_image(image, op_bbox, border_shade=128)

    scale = load_size / min(image_crop.shape[0], image_crop.shape[1])
    dest_size = (int(np.round(image_crop.shape[1] * scale)),
                 int(np.round(image_crop.shape[0] * scale)))
    image_crop = cv2.resize(image_crop,
                            dest_size,
                            interpolation=cv2.INTER_LINEAR)

    image_tensor = get_image_to_tensor()(image_crop)[None]
    return op_bbox, image_tensor


def load_image(image, load_size=512):
    image_bgr = cv2.imread(image)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    center = np.array([image.shape[1] // 2, image.shape[0] // 2])
    half_sz = np.array((max(image.shape[0], image.shape[1]) - 1) // 2 + 1)
    tlcorner = center - half_sz
    bbox = np.array([tlcorner[0], tlcorner[1], half_sz * 2, half_sz * 2])
    image = crop_image(image, bbox, border_shade=128)
    image = cv2.resize(image, (load_size, load_size),
                       interpolation=cv2.INTER_LINEAR)
    image = get_image_to_tensor()(image)[None]
    return bbox, image
