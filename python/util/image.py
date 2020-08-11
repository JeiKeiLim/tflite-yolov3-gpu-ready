import tensorflow as tf
import numpy as np
from util import data_aug
import copy


def image_augmentation_only(img, saturation=(.7, 1.3), brightness=0.2, contrast=(0.8, 1.2), hue=0.15, quality=(1, 100)):
    aug_methods = [
        lambda x: tf.image.random_saturation(x, saturation[0], saturation[1]),
        lambda x: tf.image.random_hue(img, hue),
        lambda x: tf.image.random_contrast(img, contrast[0], contrast[1]),
        lambda x: tf.image.random_brightness(img, brightness),
        lambda x: tf.image.random_jpeg_quality(img, quality[0], quality[1])
                   ]

    np.random.shuffle(aug_methods)
    for aug_method in aug_methods:
        img = aug_method(img)

    return np.array(img)


def image_augmentation_with_box(img, boxes, crop_scale=0.5, p_flip=0.5):
    box = copy.copy(boxes)

    aug_methods = [
        lambda x, y: data_aug.RandomCrop(alpha=crop_scale, beta=crop_scale)(x, y),
        lambda x, y: data_aug.RandomFlip(p=p_flip)(x, y)
    ]

    np.random.shuffle(aug_methods)
    for aug_method in aug_methods:
        img, box = aug_method(img, box)

    return img, box


def image_augmentation(img, boxes, saturation=(.7, 1.3), brightness=0.2, contrast=(0.8, 1.2), hue=0.15, quality=(1, 100),
                       crop_scale=0.5, flip_p=0.5, target_size=None):
    box = copy.copy(boxes)

    img = image_augmentation_only(img, saturation=saturation, brightness=brightness, contrast=contrast, hue=hue, quality=quality)
    img, box = image_augmentation_with_box(img, box, crop_scale=crop_scale, p_flip=flip_p)

    img = np.array(img)

    if target_size is not None:
        img, box = data_aug.Resize(target_size[0], target_size[1])(img, box)

    return img, box


def rotate_traffic_light(img, boxes, p_rotate=0.7, rotate_ratio_threshold=1.8, traffic_light_idx=3, margin=0.1):
    margin_rate = margin
    for i, box in enumerate(boxes):
        if box[4] == traffic_light_idx:
            x1, y1, x2, y2 = box[:4]
            w, h = (x2-x1), (y2-y1)

            if (h / w) < rotate_ratio_threshold:
                continue
            if np.random.rand() > p_rotate:
                continue

            margin = max(w * margin_rate, h * margin_rate)
            print(f"w: {w:.2f}, h: {h:.2f}, margin: {margin:.2f}")

            crop_x1, crop_x2, crop_y1, crop_y2 = x1, x2, y1, y2
            """
            Crop with margin of the bound box
            """
            crop_x1 = max(crop_x1 - margin, 0)
            crop_x2 = min(crop_x2 + margin, img.shape[1])
            crop_y1 = max(crop_y1 - margin, 0)
            crop_y2 = min(crop_y2 + margin, img.shape[0])

            if (crop_x2 - crop_x1) < 1 or (crop_y2 - crop_y1) < 1:
                continue

            crop_xs = (crop_x1, crop_x2)
            crop_ys = (crop_y1, crop_y2)

            crop_x1 = np.round(crop_x1).astype(np.int)
            crop_x2 = np.round(crop_x2).astype(np.int)
            crop_y1 = np.round(crop_y1).astype(np.int)
            crop_y2 = np.round(crop_y2).astype(np.int)
            margin = np.round(margin).astype(np.int)

            crop_img = img[crop_y1:crop_y2+1, crop_x1:crop_x2+1, :]
            crop_img = crop_img[:, ::-1, :]
            rot_img = np.swapaxes(crop_img, 0, 1).astype(np.uint8)
            crop_w, crop_h = rot_img.shape[:2]

            """
            Filling up an empty space where rotating objects leave by mean values of [1~8]
            [1] [2] [3]
            [4][Obj][5]
            [6] [7] [8]
            """
            mean_img = []
            for iy in range(max(crop_y1-crop_h-margin, 0), min(crop_y1+crop_h+margin, img.shape[0]-crop_h), crop_h):
                for ix in range(max(crop_x1-crop_w-margin, 0), min(crop_x1+crop_w+margin, img.shape[1]-crop_w), crop_w):
                    mean_img.append(img[iy:iy+crop_h, ix:ix+crop_w, :])

            mean_img = np.array(mean_img).mean(axis=0).astype(np.uint8)
            img[crop_y1:crop_y2+1, crop_x1:crop_x2+1, :] = mean_img

            """
            Paste rotated object into original img
            """
            n_y1 = crop_y1 + (crop_w // 2)
            n_y2 = min(n_y1 + crop_w, img.shape[0])
            n_x1 = max(crop_x1 - (crop_h // 2), 0)
            n_x2 = min(n_x1 + crop_h, img.shape[1])

            img[n_y1:n_y2, n_x1:n_x2, :] = rot_img[:n_y2-n_y1, :n_x2-n_x1, :]

            # 0: x1, 1: y1, 2: x2, 3: y2
            boxes[i, 1] = n_y1 - (crop_xs[0]-x1)
            boxes[i, 3] = n_y2 - (crop_xs[1]-x2)
            boxes[i, 0] = n_x1 - (crop_ys[0]-y1)
            boxes[i, 2] = n_x2 - (crop_ys[1]-y2)

    return img, boxes
