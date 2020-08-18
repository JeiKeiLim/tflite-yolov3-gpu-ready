import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from yolov3 import yolo_util
from util import prettyjson, image_augmentation, rotate_traffic_light
import tensorflow as tf
from util import data_aug


class BDD100kDataset:
    LOG_HEAD = "[BDD100k]"

    def __init__(self, config, load_anchor=True, verbose=1, prune_classes=True):
        assert config is not None

        self.verbose = verbose
        self.config = config

        self.root = config['dataset_path']
        self.use_classes = self.config['use_classes']
        self.class_ids = {idx: self.use_classes[idx] for idx in range(len(self.use_classes))}
        self.num_classes = len(self.use_classes)

        self.log("Loading label ...", level=0)

        with open("{}/labels/bdd100k_labels_images_train.json".format(self.root), 'r') as f:
            self.train_label = json.load(f)
        with open("{}/labels/bdd100k_labels_images_val.json".format(self.root), 'r') as f:
            self.val_label = json.load(f)

        self.log("Done loading label!", level=0)

        if prune_classes:
            self.train_label = self.get_only_use_classes_label(type="train")
            self.val_label = self.get_only_use_classes_label(type="val")

        if load_anchor:
            self.anchors = np.array(self.config['anchors'])
        else:
            self.log("Start anchor kmeans  ...", level=0)
            self.anchors = self.get_anchor_kmeans(type="train", n_cluster=9)

            self.config['anchors'] = self.anchors.tolist()
            self.log("Done anchor kmeans!", level=0)

        with open(self.config['self_path'], 'w') as f:
            f.write(prettyjson(self.config, maxlinelength=20))

    def log(self, msg, level=0):
        if self.verbose > level:
            print("{} {}".format(BDD100kDataset.LOG_HEAD, msg))

    def make_line_annotation(self, type="train"):
        if type == "train":
            label_info = self.train_label
        else:
            label_info = self.val_label

        line_annot = [(labels['name'], label['category'],
                       label['box2d']['x1'], label['box2d']['y1'],
                       label['box2d']['x2'], label['box2d']['y2'],
                       label['class_id'])
                      for labels in label_info for label in labels['labels']]
        line_annot = np.array(line_annot)

        bbox_annot = line_annot[:, 2:].astype('float32')
        file_annot = line_annot[:, :2]

        return bbox_annot, file_annot

    def get_anchor_kmeans(self, type="train", n_cluster=9):
        bboxes, file_info = self.make_line_annotation(type=type)
        width_heights = np.stack([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).T

        clustered_anchors = yolo_util.kmeans(width_heights, n_cluster)

        img, label = self.get_sample_data(0)
        target_ratio = self.config['input_shape'][::-1] / np.array(img.size)

        clustered_anchors = clustered_anchors * target_ratio
        clustered_anchors = clustered_anchors.astype('float16')
        clustered_anchors = clustered_anchors[np.lexsort(clustered_anchors.T)]

        return clustered_anchors

    def get_only_use_classes_label(self, type="train"):
        if type == "train":
            label_info = self.train_label
        else:
            label_info = self.val_label

        img, box = self.get_sample_data(0, prefix=type)
        img_size = img.size # w, h

        new_label = []
        for labels in label_info:
            do_i_use = False
            this_label = []
            for label in labels['labels']:
                if 'box2d' not in label.keys():
                    continue

                # Excluding labels with attribute conditions.
                if label['category'] in self.config['exclude_class_with_attributes'].keys():
                    exclude_attrs = dict(self.config['exclude_class_with_attributes'][label['category']])
                    exclude_type = exclude_attrs.pop("exclude_condition")
                    condition_list = []

                    for ex_attr, ex_value in exclude_attrs.items():
                        condition_list.append(label['attributes'][ex_attr] == ex_value)

                    if (exclude_type == "or" and np.sum(condition_list) > 0) or \
                            (exclude_type == "and" and np.sum(condition_list) == len(condition_list)):
                        continue

                # Excluding labels with box size ratio conditions.
                if label['category'] in self.config['exclude_class_with_box_ratio'].keys():
                    ratio_threshold = self.config['exclude_class_with_box_ratio'][label['category']]
                    x1, x2, y1, y2 = label['box2d']['x1'], label['box2d']['x2'], label['box2d']['y1'], label['box2d']['y2']
                    w, h = (x2-x1), (y2-y1)
                    ratio = np.sqrt(img_size[0]*img_size[1]) / np.sqrt(w*h)
                    if ratio < ratio_threshold:
                        continue

                if label['category'] in self.use_classes:
                    label['class_id'] = np.argmax(label['category'] == np.array(self.use_classes))
                    this_label.append(label)

                    if len(self.config['must_include_classes']) == 0:
                        do_i_use = True
                    elif label['category'] in self.config['must_include_classes']:
                        do_i_use = True

            if do_i_use and len(this_label) > 0:
                new_label.append({'name': labels['name'],
                                  'attributes': labels['attributes'],
                                  'labels': this_label})

        return new_label

    def get_sample_data(self, idx, prefix="train"):
        if prefix == "train":
            label = self.train_label[idx]
        else:
            label = self.val_label[idx]

        f_path_ = "{}/images/100k/{}/{}".format(self.root, prefix, label['name'])
        img = Image.open(f_path_)
        label_info = label['labels']

        return img, label_info

    def label_to_bbox(self, label):
        return np.array([[label[i]['box2d']['x1'], label[i]['box2d']['y1'],
                         label[i]['box2d']['x2'], label[i]['box2d']['y2'],
                         label[i]['class_id']]
                        for i in range(len(label))])

    def plot_label(self, idx, prefix="train", show_attr=False, figscale=20.0, show_bbox_ratio=False, title=""):
        img, labels = self.get_sample_data(idx, prefix=prefix)
        bboxes = self.label_to_bbox(labels)

        attrs = self.get_attributes_msg(labels) if show_attr else None

        # attrs = [", ".join(x for x in [key if type(value) == bool and value is True
        #                                else value if type(value) == str and value != 'none'
        #                                else None
        #                                for key, value in label['attributes'].items()] if x)
        #                                for label in labels] if show_attr else None

        self.plot_bboxes(img, bboxes, attributes=attrs, figscale=figscale, show_bbox_ratio=show_bbox_ratio, title=title)

    def get_attributes_msg(self, labels):
        attrs = [", ".join(x for x in [key if type(value) == bool and value is True
                                       else value if type(value) == str and value != 'none' else None
                                       for key, value in label['attributes'].items()] if x)
                 for label in labels]
        return attrs

    def plot_bboxes(self, img, bboxes, attributes=None, figscale=20.0, ax=None, show_bbox_ratio=False, title=""):
        colors = [(0.8, 0.1, 0.1, 0.6),
                     (0.1, 0.8, 0.1, 0.6),
                     (0.1, 0.1, 0.8, 0.6),
                     (0.8, 0.1, 0.8, 0.6),
                     (0.8, 0.8, 0.1, 0.6),
                     (0.1, 0.8, 0.8, 0.6),
                     (0.3, 0.8, 0.3, 0.6)]
        img_w, img_h = (img.shape[1], img.shape[0]) if type(img) == np.ndarray else img.size

        fig_w = (img_w/(img_w+img_h)) * figscale
        fig_h = (img_h/(img_w+img_h)) * figscale

        if ax is None:
            fig, fig_ax = plt.subplots(1, figsize=(fig_w, fig_h))
        else:
            fig_ax = ax

        if type(img) == np.ndarray:
            img_size = np.sqrt(img.shape[0] * img.shape[1])
        else:
            img_size = np.sqrt(img.size[0] * img.size[1])

        fig_ax.imshow(img)
        for i, box in enumerate(bboxes):
            if np.sum(box) == 0:
                continue

            color = colors[int(box[4]) % len(colors)]
            w = box[2] - box[0]
            h = box[3] - box[1]

            box_ratio = np.sqrt(w*h) / img_size

            rect = patches.Rectangle((box[0], box[1]), w, h,
                                     linewidth=min(max(box_ratio*20, 1), 2),
                                     edgecolor=color[:3] + (0.8, ),
                                     facecolor='none')
            fig_ax.add_patch(rect)
            txt_msg = "{}".format(self.class_ids[box[4]])
            if attributes:
                txt_msg += "({})".format(attributes[i]) if attributes[i] != "" else ""
            if show_bbox_ratio:
                txt_msg += ", BOX({:03.1f}%)".format(box_ratio*100)

            random_x_loc = np.random.uniform(box[0], box[2]*0.9)
            fig_ax.text(random_x_loc, box[1]-8, txt_msg,
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        backgroundcolor=color,
                        color=(1.0, 1.0, 1.0, 0.9),
                        fontsize=(min(max(box_ratio*120, 6), 13)))
        fig_ax.axis('off')
        if title != "":
            fig_ax.set_title(title)

        if ax is None:
            fig.show()
        else:
            return fig_ax

    def get_preprocessed_data(self, idx, input_shape, max_boxes=20, type="train", augment=True,
                              p_rotate_trafficlight=0.7, rotate_trafficlight_ratio_threshold=1.8, trafficlight_idx=3):
        img, label = self.get_sample_data(idx, prefix=type)
        img = np.array(img)
        box = self.label_to_bbox(label)

        if augment:
            img, box = image_augmentation(img, box, target_size=(input_shape[1], input_shape[0]))
            img, box = rotate_traffic_light(img, box, p_rotate=p_rotate_trafficlight,
                                            rotate_ratio_threshold=rotate_trafficlight_ratio_threshold,
                                            traffic_light_idx=trafficlight_idx)
        else:
            img, box = data_aug.Resize(input_shape[1], input_shape[0])(img, box)

        out_box = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box = box[:max_boxes]
            out_box[:len(box)] = box

        img = img / 255

        return img, out_box

    def data_generator(self, batch_size, input_shape, type="train", shuffle=True, augment=True):
        i = 0
        labels = self.train_label if type == "train" else self.val_label
        n = len(labels)

        idx = np.arange(len(labels))
        if shuffle:
            np.random.shuffle(idx)

        while True:
            image_data = []
            box_data = []

            for b in range(batch_size):
                image, box = self.get_preprocessed_data(idx[i], input_shape, type=type, augment=augment,
                                                        p_rotate_trafficlight=self.config['rotate_trafficlight']['p'],
                                                        rotate_trafficlight_ratio_threshold=self.config['rotate_trafficlight']['ratio_threshold'],
                                                        trafficlight_idx=np.argwhere(np.array(list(self.class_ids.values())) == 'traffic light').flatten()[0])
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
                if i == 0 and shuffle:
                    np.random.shuffle(idx)

            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = yolo_util.preprocess_true_boxes(box_data, input_shape, self.anchors, self.num_classes, grid_ratio=self.config['grid_ratio'])

            yield [image_data, *y_true], np.zeros(batch_size), box_data
