from mobilenet import MobileNetV1
import numpy as np
import tensorflow as tf
import tflitegpu as tfgpu
from yolov3 import yolo_util
from tensorflow.python.ops import control_flow_ops
from deprecated import deprecated
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YoloJK:
    def __init__(self, yolo_config):
        self.config = yolo_config

        self.anchors = np.array(self.config['anchors'])
        self.n_filters = self.config['n_filters']
        self.n_anchors = len(self.anchors) // 3
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if self.n_anchors == 3 else [[3, 4, 5], [1, 2, 3]]
        self.n_classes = len(self.config['class_ids'])
        self.input_shape = self.config['input_shape']
        self.model = None
        self.loss_model = None
        self.prediction_model = None

    def build_model(self):
        if self.model is not None:
            return self.model

        model = MobileNetV1(input_shape=self.input_shape,
                            n_classes=0,
                            alpha=self.config['alpha']).build_model()

        x1 = model.get_layer("l12_02_ReLU").output
        x2 = model.get_layer("l10_02_ReLU").output
        x3 = model.get_layer("l04_02_ReLU").output

        x, y1 = self.build_last_layer(x1, self.n_filters[0])
        x = self.upsample_layer(x, x2, self.n_filters[1])

        x, y2 = self.build_last_layer(x, self.n_filters[1])
        x = self.upsample_layer(x, x3, self.n_filters[2])

        x, y3 = self.build_last_layer(x, self.n_filters[2])

        self.model = tf.keras.models.Model(model.input, [y1, y2, y3])

        return self.model

    def load_weights(self, path="", load_train=False):
        if self.model is None:
            self.build_model()

        if path == "":
            path = self.config['train']['load_weight_path'] if load_train else self.config['weight_path']

        try:
            self.model.load_weights(path)
            print(f"Load wegiths from {path}")
        except:
            print(f"Failed loading wegiths from {path}")
            pass

        return self.model

    def upsample_layer(self, x, x2, n_filter):
        x = tfgpu.Conv2DBNRelu(n_filter, kernel_size=1, strides=(1, 1), padding='same', use_bias=False, relu_max=6.)(x)
        x = tfgpu.UpSampling2D(2)(x)
        x = tf.keras.layers.Concatenate()([x, x2])

        return x

    def build_last_layer(self, x, n_filters):
        for i in range(3):
            x = tf.keras.layers.Conv2D(n_filters, kernel_size=1, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(axis=-1)(x)
            x = tf.keras.layers.ReLU(6.)(x)
            x = tfgpu.SeparableBatchConv2D(n_filters, kernel_size=3,
                                           strides=(1, 1), padding='same',
                                           use_bias=False, activation=None)(x)
            x = tf.keras.layers.ReLU(6.)(x)

        y = tfgpu.SeparableBatchConv2D(n_filters, kernel_size=3, strides=(1, 1), use_bias=False, activation=None)(x)
        y = tf.keras.layers.Conv2D(self.n_anchors * (5 + self.n_classes), 1, padding='same', use_bias=False)(y)

        return x, y

    def build_prediction_model(self, image_shape=None, max_boxes=20, score_threshold=0.6, iou_threshold=.5):
        if self.prediction_model is not None:
            return self.prediction_model

        # if image_shape is None:
        #     image_shape = self.input_shape[:2]

        decoder = YoloDecoder(self.config)
        pred_out = decoder.evaluation_layer(self.model.output, image_shape=image_shape, max_boxes=20, score_threshold=0.6, iou_threshold=.5)

        # pred_out = yolo_eval(self.model.output, self.anchors, self.n_classes, image_shape,
        #                      max_boxes=max_boxes,
        #                      score_threshold=score_threshold,
        #                      iou_threshold=iou_threshold)

        self.prediction_model = tf.keras.models.Model(self.model.input, pred_out)

        return self.prediction_model

    def build_loss_model(self):
        if self.loss_model is not None:
            return self.loss_model

        loss = YoloLoss(self.config, self.model, name="yolo_loss")
        self.loss_model = loss.loss_model

        return self.loss_model


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, yolo_config, yolo_model, reduction=tf.keras.losses.Reduction.AUTO, name=None, ignore_threshold=0.5):
        super(YoloLoss, self).__init__(reduction=reduction, name=name)
        self.yolo_config = yolo_config

        self.anchors = np.array(yolo_config['anchors'])
        self.n_anchors = len(self.anchors) // 3
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if self.n_anchors == 3 else [[3, 4, 5], [1, 2, 3]]

        self.ignore_threshold = ignore_threshold
        self.n_classes = len(self.yolo_config['class_ids'])
        self.grid_ratio = self.yolo_config['grid_ratio']
        self.input_shape = self.yolo_config['input_shape'][:2]
        self.yolo_decoder = YoloDecoder(yolo_config)

        self.yolo_model = yolo_model
        self.loss_model = self.build_loss_model()

    def build_loss_model(self):
        y_true_inputs = [tf.keras.layers.Input(shape=(self.input_shape[0] // self.grid_ratio[str(l)],
                                      self.input_shape[1] // self.grid_ratio[str(l)], 3, self.n_classes + 5))
                  for l in range(self.n_anchors)]

        loss_lambda = tf.keras.layers.Lambda(self.yolo_loss, output_shape=(1,), name='yolo_loss')(
            [*self.yolo_model.output, *y_true_inputs])
        self.loss_model = tf.keras.models.Model([self.yolo_model.input, *y_true_inputs], loss_lambda)

        return self.loss_model

    def call(self, y_true, y_pred):
        pass

    def yolo_loss(self, args):
        yolo_outputs = args[:self.n_anchors]
        y_true = args[self.n_anchors:]

        input_shape = tf.cast(tf.shape(y_true[0])[1:3] * 32, y_true[0].dtype)
        grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], y_true[l].dtype) for l in range(self.n_anchors)]

        m = tf.shape(yolo_outputs[0])[0]

        loss = 0.0
        for l in range(self.n_anchors):
            loss += self.loss_layer(y_true[l], yolo_outputs[l], input_shape, grid_shapes[l], m, l)

        return loss

    def loss_layer(self, y_true, y_pred, input_shape, grid_shapes, m, l):
        object_mask = y_true[..., 4:5]
        true_class_probs = y_true[..., 5:]

        grid, raw_pred, pred_xy, pred_wh = self.yolo_decoder.decode_layer(y_pred, l, calc_loss=True)
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[..., :2] * grid_shapes[::-1] - grid
        raw_true_wh = tf.math.log(y_true[..., 2:4] / self.anchors[self.anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = tf.keras.backend.switch(object_mask, raw_true_wh, tf.zeros_like(raw_true_wh)) # avoid log(0)=-inf

        box_loss_scale = 2 - y_true[..., 2:3]*y_true[..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(y_true.dtype, size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, tf.bool)

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[b,...,0:4], object_mask_bool[b,...,0])
            iou = yolo_util.box_iou(pred_box[b], true_box)
            best_iou = tf.keras.backend.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou<self.ignore_threshold, true_box.dtype))
            return b+1, ignore_mask

        _, ignore_mask = control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * tf.expand_dims(tf.keras.losses.binary_crossentropy(raw_true_xy, raw_pred[..., :2], from_logits=True), axis=-1)
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh-raw_pred[..., 2:4])
        confidence_loss = object_mask * tf.expand_dims(tf.keras.losses.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True), axis=-1)+ \
            (1-object_mask) * tf.expand_dims(tf.keras.losses.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True), axis=-1) * ignore_mask
        class_loss = object_mask * tf.expand_dims(tf.keras.losses.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True), axis=-1)

        m = tf.cast(m, y_pred.dtype)

        xy_loss = tf.keras.backend.sum(xy_loss) / m
        wh_loss = tf.keras.backend.sum(wh_loss) / m
        confidence_loss = tf.keras.backend.sum(confidence_loss) / m
        class_loss = tf.keras.backend.sum(class_loss) / m

        return xy_loss + wh_loss + confidence_loss + class_loss


class YoloDecoder:
    def __init__(self, yolo_config):
        self.config = yolo_config
        self.anchors = np.array(yolo_config['anchors'])
        self.n_anchors = len(self.anchors) // 3
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if self.n_anchors == 3 else [[3, 4, 5], [1, 2, 3]]
        self.class_ids = yolo_config['class_ids']
        self.n_classes = len(yolo_config['class_ids'])
        self.input_shape = yolo_config['input_shape'][:2]

    def plot_result(self, boxes, scores, class_idx, image_size=None, img=None, img_shape=None,
                    box_colors=None, figscale=20.0, ax=None, show=True, title=""):
        """

        :param boxes: (n, 4) array. (x_min, y_min, x_max, y_max)
        :param scores: (n, ) array
        :param class_idx: (n, ) array
        :param image_size: (height, width) tuple or array
        :param img: (height, width, channel) array
        :param box_colors:
        :param figscale:
        :param ax:
        :param show:
        :param title:
        :param save_path:
        :return:
        """
        assert (ax is None and (image_size is not None or img is not None)) or ax is not None
        assert img is not None or img_shape is not None

        if img_shape is None:
            img_shape = img.shape

        if box_colors is None:
            box_colors = [(0.8, 0.1, 0.1, 0.6),
                      (0.1, 0.8, 0.1, 0.6),
                      (0.1, 0.1, 0.8, 0.6),
                      (0.8, 0.1, 0.8, 0.6),
                      (0.8, 0.8, 0.1, 0.6),
                      (0.1, 0.8, 0.8, 0.6),
                      (0.3, 0.8, 0.3, 0.6)]

        if ax is None:
            if image_size is None:
                image_size = img.shape[:2]

            img_w, img_h = image_size[1], image_size[0]
            fig_w = (img_w / (img_w + img_h)) * figscale
            fig_h = (img_h / (img_w + img_h)) * figscale

            fig, ax = plt.subplots(1, figsize=(fig_w, fig_h))
        else:
            fig = None

        if img is not None:
            ax.imshow(img)

        for box_idx, t_box in enumerate(boxes):
            tx, ty, tx2, ty2 = t_box
            tw, th = tx2 - tx, ty2 - ty

            box_ratio = np.sqrt(tw * th) / np.sqrt(img_shape[0] * img_shape[1])
            color = box_colors[int(class_idx[box_idx]) % len(box_colors)]

            rect = patches.Rectangle((tx, ty), tw, th,
                                     linewidth=min(max(box_ratio*20, 1), 2),
                                     edgecolor=color[:3] + (0.8, ), facecolor='none'
                                     )
            ax.add_patch(rect)
            msg = "{} ({:.2f}%)".format(self.class_ids[class_idx[box_idx]], scores[box_idx]*100)

            random_x_loc = np.random.uniform(tx, tx2 * 0.9)
            ax.text(random_x_loc, ty, msg,
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    backgroundcolor=color,
                    color=(1.0, 1.0, 1.0, 0.9),
                    fontsize=(min(max(box_ratio * 120, 6), 13))
                    )

        ax.axis('off')

        if title != "":
            ax.set_title(title)

        if show and fig:
            fig.show()

        if fig is not None:
            return fig, ax
        else:
            return ax

    def decode_layer(self, feats, l, calc_loss=False):
        """Convert final layer features to bounding box parameters."""
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = tf.reshape(tf.constant(self.anchors[self.anchor_mask[l]], dtype=feats.dtype), [1, 1, 1, self.n_anchors, 2])

        input_shape = tf.constant(self.input_shape)

        grid_shape = tf.shape(feats)[1:3]  # height, width
        grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]),
                        [1, grid_shape[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]),
                        [grid_shape[0], 1, 1, 1])
        grid = tf.cast(tf.concat([grid_x, grid_y], axis=-1), feats.dtype)

        feats = tf.reshape(
            feats, [-1, grid_shape[0], grid_shape[1], self.n_anchors, self.n_classes + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], feats.dtype)
        box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], feats.dtype)

        box_confidence = tf.sigmoid(feats[..., 4:5])
        box_class_probs = tf.sigmoid(feats[..., 5:])

        if calc_loss:
            return grid, feats, box_xy, box_wh

        return box_xy, box_wh, box_confidence, box_class_probs

    def correct_boxes(self, box_xy, box_wh, image_shape):
        '''Get corrected boxes'''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape = tf.cast(self.input_shape, box_yx.dtype)
        image_shape = tf.cast(image_shape, box_yx.dtype)
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)

        # Scale boxes back to original image shape.
        boxes *= tf.concat([image_shape, image_shape], axis=0)
        return boxes

    def evaluation_layer(self, yolo_outs, image_shape=None, max_boxes=20, score_threshold=.6, iou_threshold=.5):
        if image_shape is None:
            image_shape = self.input_shape

        boxes = []
        box_scores = []

        for l in range(len(yolo_outs)):
            xy, wh, conf, score = self.decode_layer(yolo_outs[l], l, calc_loss=False)
            box = self.correct_boxes(xy, wh, image_shape)
            score = conf * score

            box = tf.reshape(box, [-1, 4])
            score = tf.reshape(score, [-1, self.n_classes])
            boxes.append(box)
            box_scores.append(score)

        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []

        for c in range(self.n_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=iou_threshold)
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)

        return boxes_, scores_, classes_

    def decode_from_ground_truth(self, feats, image_shape):
        """

        :param feats: Ground truth feature map from data generator (batch_size x height x width x (5+n_classes)
        :param image_shape: Target Image Size (width x height)
        :return:
        """

        def decode_feat(feat):
            batch_size = feat.shape[0]
            out_h = feat.shape[1]
            out_w = feat.shape[2]

            feat = np.reshape(feat, (batch_size, out_h, out_w, self.n_anchors, 5 + self.n_classes))

            raw_xy, raw_wh, raw_conf, raw_prob = np.split(feat, [2, 4, 5], axis=-1)

            raw_xy = raw_xy.reshape(-1, 2)
            raw_wh = raw_wh.reshape(-1, 2)

            min_xy = (raw_xy - (raw_wh / 2.0)) * image_shape
            max_xy = (raw_xy + (raw_wh / 2.0)) * image_shape

            raw_conf = raw_conf.reshape(-1, 1)
            raw_prob = raw_prob.reshape(-1, self.n_classes)

            mask = np.where((raw_conf > 0.5).flatten())
            min_xy = min_xy[mask]
            max_xy = max_xy[mask]
            raw_conf = raw_conf[mask].flatten()
            raw_prob = raw_prob[mask]
            classes = np.argmax(raw_prob, axis=1)
            raw_xyxy = np.concatenate([min_xy, max_xy], axis=-1)

            return raw_xyxy, raw_conf, classes

        outs = [decode_feat(feat_) for feat_ in feats]
        raw_xyxy = np.concatenate([out[0] for out in outs])
        raw_conf = np.concatenate([out[1] for out in outs])
        classes = np.concatenate([out[2] for out in outs])

        return raw_xyxy, raw_conf, classes

    @deprecated(reason="TFLite does not support ops yet.")
    def decode_layer_tflite(self, feats, l):
        # anchors_tensor = tf.reshape(tf.constant(self.anchors[self.anchor_mask[l]], dtype=feats.dtype),
        #                             [-1, 1, self.n_anchors, 2])
        # n_anchor = tf.constant(self.n_anchors)
        n_classes = tf.constant(self.n_classes)
        n_channel = n_classes + 5

        # grid_shape = tf.shape(feats)[1:3]  # height, width
        grid_y = tf.tile(tf.reshape(tf.range(0, feats.shape[1], dtype=feats.dtype), [-1, 1, 1]),
                         [1, feats.shape[2], 1])
        grid_x = tf.tile(tf.reshape(tf.range(0, feats.shape[2], dtype=feats.dtype), [1, -1, 1]),
                         [feats.shape[1], 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.tile(tf.expand_dims(grid, axis=0), [tf.shape(feats)[0], 1, 1, 1])

        # feats0, feats1, feats2 = tf.split(feats, [n_channel, n_channel, n_channel], axis=-1)

        feats0 = feats[:, :, :, :n_channel]
        feats1 = feats[:, :, :, n_channel:(n_channel*2)]
        feats2 = feats[:, :, :, (n_channel*2):(n_channel*3)]

        # Adjust preditions to each spatial grid point and anchor size.
        xy_div = tf.cast(feats.shape[1:3][::-1], feats.dtype)
        wh_div = tf.cast(self.input_shape[::-1], feats.dtype)

        # anchor0 = tf.cast(self.anchors[self.anchor_mask[l]][0], feats.dtype)
        # anchor1 = tf.cast(self.anchors[self.anchor_mask[l]][1], feats.dtype)
        # anchor2 = tf.cast(self.anchors[self.anchor_mask[l]][2], feats.dtype)

        box_xy0 = tf.sigmoid(feats0[:, :, :, 0:2])
        box_xy1 = tf.sigmoid(feats1[:, :, :, 0:2])
        box_xy2 = tf.sigmoid(feats2[:, :, :, 0:2])
        # box_xy1 = (tf.sigmoid(feats1[:, :, :, 0:2]) + grid) / xy_div
        # box_xy2 = (tf.sigmoid(feats2[:, :, :, 0:2]) + grid) / xy_div

        box_wh0 = feats0[:, :, :, 2:4]
        box_wh1 = feats1[:, :, :, 2:4]
        box_wh2 = feats2[:, :, :, 2:4]
        # box_wh0 = tf.exp(feats0[:, :, :, 2:4]) * anchor0 / wh_div
        # box_wh1 = tf.exp(feats1[:, :, :, 2:4]) * anchor1 / wh_div
        # box_wh2 = tf.exp(feats2[:, :, :, 2:4]) * anchor2 / wh_div

        box_confidence0 = tf.sigmoid(feats0[:, :, :, 4:5])
        box_confidence1 = tf.sigmoid(feats1[:, :, :, 4:5])
        box_confidence2 = tf.sigmoid(feats2[:, :, :, 4:5])

        box_class_probs0 = tf.sigmoid(feats0[:, :, :, 5:])
        box_class_probs1 = tf.sigmoid(feats1[:, :, :, 5:])
        box_class_probs2 = tf.sigmoid(feats2[:, :, :, 5:])

        # box_xy = tf.concat([box_xy0, box_xy1, box_xy2], axis=-1)
        # box_wh = tf.concat([box_wh0, box_wh1, box_wh2], axis=-1)
        # box_confidence = tf.concat([box_confidence0, box_confidence1, box_confidence2], axis=-1)
        # box_class_probs = tf.concat([box_class_probs0, box_class_probs1, box_class_probs2], axis=-1)

        feats = tf.concat([box_xy0, box_wh0, box_confidence0, box_class_probs0, box_xy1, box_wh1, box_confidence1, box_class_probs1, box_xy2, box_wh2, box_confidence2, box_class_probs2], axis=-1)
        return feats

