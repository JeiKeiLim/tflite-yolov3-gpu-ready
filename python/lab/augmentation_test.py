from gonogo_conf import get_arg_parser
from dataset.bdd100k_dataset import BDD100kDataset
import numpy as np
import matplotlib.pyplot as plt
from util import image_augmentation, rotate_traffic_light
import copy


def plot_augmentation(bdd100k_dataset, img_idx=None, prefix="train",
                      saturation=(.7, 1.3), brightness=0.2,
                      contrast=(0.8, 1.2), hue=0.15, quality=(1, 100),
                      crop_scale=0.5, flip_p=0.5):
    plt.figure(figsize=(50, 30))

    img_idx = np.random.randint(0, len(bdd100k_dataset.train_label)) if img_idx is None else img_idx

    img, label = bdd100k_dataset.get_sample_data(img_idx, prefix=prefix)
    img = np.array(img)
    ax = plt.subplot(5, 5, 1)
    boxes = bdd100k_dataset.label_to_bbox(label)
    attrs = bdd100k_dataset.get_attributes_msg(label)
    bdd100k_dataset.plot_bboxes(img, boxes, ax=ax, attributes=attrs, show_bbox_ratio=True)

    for i in range(2, 26):
        ax = plt.subplot(5, 5, i)

        img, label = bdd100k_dataset.get_sample_data(img_idx, prefix=prefix)
        boxes = bdd100k_dataset.label_to_bbox(label)
        img = np.array(img)

        img, boxes = image_augmentation(img, boxes, crop_scale=crop_scale, saturation=saturation, brightness=brightness,
                                 contrast=contrast, hue=hue, quality=quality, flip_p=flip_p)

        img, boxes = rotate_traffic_light(img, boxes, p_rotate=bdd100k_dataset.config['rotate_trafficlight']['p'],
                                        rotate_ratio_threshold=bdd100k_dataset.config['rotate_trafficlight']['ratio_threshold'],
                                        traffic_light_idx=np.argwhere(np.array(list(bdd100k_dataset.class_ids.values())) == 'traffic light').flatten()[0])
        attrs = bdd100k_dataset.get_attributes_msg(label)
        bdd100k_dataset.plot_bboxes(img, boxes, ax=ax, attributes=attrs, show_bbox_ratio=True)

    plt.show()

    img, label = bdd100k_dataset.get_sample_data(img_idx, prefix=prefix)
    boxes = bdd100k_dataset.label_to_bbox(label)

    return np.array(img), boxes


if __name__ == '__main__':
    parser, dataset_config, model_config, tflite_config = get_arg_parser(return_config=True)
    args = parser.parse_args()

    bdd100k = BDD100kDataset(dataset_config, load_anchor=True)

    _, _ = plot_augmentation(bdd100k, img_idx=None, prefix="train",
                      saturation=(.7, 1.3), brightness=0.2,
                      contrast=(0.8, 1.2), hue=0.15, quality=(1, 100))

    img_idx = np.random.randint(0, len(bdd100k.train_label))
    # img_idx = 17606
    # img_idx = 27892
    # img_idx = 25489

    img, label = bdd100k.get_sample_data(img_idx, prefix="train")
    img = np.array(img)
    boxes = bdd100k.label_to_bbox(label)

    img2, boxes2 = image_augmentation(copy.copy(img), copy.copy(boxes), target_size=(416, 256))
    img3, boxes3 = rotate_traffic_light(copy.copy(img2), copy.copy(boxes2), p_rotate=1.0, margin=0.1,
                                      rotate_ratio_threshold=bdd100k.config['rotate_trafficlight']['ratio_threshold'],
                                      traffic_light_idx=np.argwhere(np.array(list(bdd100k.class_ids.values())) == 'traffic light').flatten()[0])
    bdd100k.plot_bboxes(img, boxes, title=f"01. Original. (Train Set, Sample No. {img_idx:,})")
    bdd100k.plot_bboxes(img2, boxes2, title=f"02. Augmentation. (Train Set, Sample No. {img_idx:,})")
    bdd100k.plot_bboxes(img3, boxes3, title=f"03. Augmentation+Traffic Light Rotation. (Train Set, Sample No. {img_idx:,})")

