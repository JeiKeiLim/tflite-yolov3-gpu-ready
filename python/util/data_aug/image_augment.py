import numpy as np
import cv2


def box_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def filter_box(bbox, filter_box, alpha):

    mask_x = bbox[:, 0] < filter_box[2]
    mask_y = bbox[:, 1] < filter_box[3]

    bbox = bbox[np.logical_and(mask_x, mask_y)]
    box_sizes = box_area(bbox)

    x_min = np.maximum(bbox[:, 0], filter_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], filter_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], filter_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], filter_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    mask_x = bbox[:, 0] < bbox[:, 2]
    mask_y = bbox[:, 1] < bbox[:, 3]
    mask = np.logical_and(mask_x, mask_y)
    bbox = bbox[mask]
    box_sizes = box_sizes[mask]

    delta_area = ((box_sizes - box_area(bbox)) / box_sizes)
    mask = delta_area < (1 - alpha)
    bbox = bbox[mask]

    return bbox


class RandomCrop(object):
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img, bbox, keep_dims=True, keep_ratio=True, filter_alpha=0.3):
        h, w = img.shape[:2]
        crop_ratio = np.array([np.random.uniform(self.alpha, 1), np.random.uniform(self.beta, 1)])
        if keep_ratio:
            idx = np.random.randint(0, 2)
            crop_ratio[0] = crop_ratio[idx]
            crop_ratio[1] = crop_ratio[idx]

        iw, ih = ([w, h] * crop_ratio).astype('int')
        margin_w, margin_h = w-iw, h-ih

        offset_x, offset_y = np.random.randint(0, margin_w+1), np.random.randint(0, margin_h+1)

        return Crop(offset_x, offset_y, offset_x+iw, offset_y+ih)(img, bbox, keep_dims=keep_dims,  filter_alpha=filter_alpha)


class Crop(object):
    def __init__(self, x1, y1, x2, y2):
        assert x1 < x2 and y1 < y2

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __call__(self, img, bboxes, keep_dims=True, filter_alpha=0.3):
        h, w = img.shape[:2]
        tw, th = (self.x2-self.x1), (self.y2-self.y1)
        assert tw <= w and th <= h

        img = img[self.y1:self.y2, self.x1:self.x2, :]
        bboxes = filter_box(bboxes, (self.x1, self.y1, self.x2, self.y2), alpha=filter_alpha)

        if bboxes.shape[0] > 0:
            bboxes[:, 0] -= self.x1
            bboxes[:, 2] -= self.x1
            bboxes[:, 1] -= self.y1
            bboxes[:, 3] -= self.y1

        if keep_dims:
            return Resize(w, h)(img, bboxes)
        else:
            return img, bboxes


class Zoom(object):
    def __init__(self, zoom_x=1.2, zoom_y=1.2):
        assert zoom_x >= 1.0 and zoom_y >= 1.0
        self.zoom_x = zoom_x
        self.zoom_y = zoom_y

    def __call__(self, img, bboxes):
        h, w = img.shape[:2]
        width = int(w * self.zoom_x)
        height = int(h * self.zoom_y)

        return Resize(width, height)(img, bboxes)


class Resize(object):
    def __init__(self, width, height):
        self.w = width
        self.h = height

    def __call__(self, img, bboxes):
        h, w = img.shape[:2]
        scale_x = self.w/w
        scale_y = self.h/h

        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        bboxes[:, :4] *= [scale_x, scale_y, scale_x, scale_y]

        return img, bboxes


class RandomFlip(object):
    def __init__(self, p=0.5, flip_code=1):
        self.p = p
        self.flip_code = flip_code

    def __call__(self, img, bboxes):
        if np.random.rand() > self.p:
            return Flip(flip_code=self.flip_code)(img, bboxes)
        else:
            return img, bboxes


class Flip(object):
    def __init__(self, flip_code=1):
        self.flip_code = flip_code
        if flip_code > 0:
            self.flip_idx = [0, 2]
        elif flip_code  < 0:
            self.flip_idx = [0, 1, 2, 3]
        else:
            self.flip_idx = [1, 3]
        self.flip_idx = np.array(self.flip_idx)

    def __call__(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))

        if self.flip_code > 0:
            img = img[:, ::-1, :]
        elif self.flip_code < 0:
            img = img[::-1, ::-1, :]
        else:
            img = img[::-1, :, :]

        bboxes[:, self.flip_idx] += 2 * (img_center[self.flip_idx] - bboxes[:, self.flip_idx])

        box_w = np.abs(bboxes[:, 0] - bboxes[:, 2])
        box_h = np.abs(bboxes[:, 1] - bboxes[:, 3])
        box_adj = [-box_w, -box_h, box_w, box_h]
        for idx in self.flip_idx:
            bboxes[:, idx] += box_adj[idx]

        return img, bboxes
