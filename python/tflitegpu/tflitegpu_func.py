import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod


class GlobalAveragePooling2D:
    def __init__(self):
        pass

    def __call__(self, layer):
        h, w = layer.shape[1:3]
        layer = tf.keras.layers.AveragePooling2D((h, w), padding='valid')(layer)
        layer = tf.keras.layers.Flatten()(layer)

        return layer


class Conv2DAbstract(ABC):
    def __init__(self, filters, kernel_size=1, strides=(3, 3), padding='same', use_bias=False, prefix=None,
                 activation=tf.keras.layers.ReLU):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.prefix = prefix
        self.activation = activation

    def name(self, postfix):
        return f"{self.prefix}{postfix}" if self.prefix is not None else None

    @abstractmethod
    def __call__(self, layer):
        raise NotImplemented


class SeparableBatchConv2D(Conv2DAbstract):
    def __init__(self, *args, **kwargs):
        Conv2DAbstract.__init__(self, *args, **kwargs)

    def __call__(self, layer, **kwargs_activation):
        layer = tf.keras.layers.DepthwiseConv2D(self.kernel_size, strides=self.strides, padding=self.padding,
                                                       use_bias=self.use_bias,
                                                       name=self.name("_DConv"))(layer)
        layer = tf.keras.layers.Conv2D(self.filters, kernel_size=1, strides=(1, 1), padding='SAME', use_bias=self.use_bias,
                                       name=self.name("_Conv1x1"))(layer)

        layer = tf.keras.layers.BatchNormalization(name=self.name("_BNorm"))(layer)
        if self.activation is not None:
            layer = self.activation(**kwargs_activation, name=self.name("_ReLU"))(layer)

        return layer


class Resize:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, layer):
        resize = (np.array(layer.shape[1:3]) * self.scale).astype(np.int32)

        return tf.image.resize(layer, resize)


class DownSampling2D(Resize):
    def __init__(self, scale):
        assert 0.0 < scale < 1.0
        Resize.__init__(self, scale)


class UpSampling2D(Resize):
    def __init__(self, scale):
        assert scale > 1.0
        Resize.__init__(self, scale)


class MobileNetV1Block:
    def __init__(self, filters, reduce_size=False, prefix=None):
        assert len(filters) == 2

        self.filter1 = filters[0]
        self.filter2 = filters[1]
        self.reduce_size = reduce_size
        self.prefix = prefix

    def __call__(self, layer):
        strides = (2, 2) if self.reduce_size else (1, 1)
        layer = SeparableBatchConv2D(self.filter1, (3, 3), strides, 'same', False,
                                     prefix=f"{self.prefix}_01" if self.prefix else None)(layer)
        layer = SeparableBatchConv2D(self.filter2, (1, 1), (1, 1), 'same', False,
                                     prefix=f"{self.prefix}_02" if self.prefix else None)(layer)

        return layer


class Conv2DBNRelu(Conv2DAbstract):
    def __init__(self, *args, relu_max=None, **kwargs):
        Conv2DAbstract.__init__(self, *args, **kwargs)
        self.relu_max = relu_max

    def __call__(self, layer):
        layer = tf.keras.layers.Conv2D(self.filters, self.kernel_size,
                                       strides=self.strides, padding=self.padding,
                                       use_bias=self.use_bias, name=self.name("_Conv"))(layer)
        layer = tf.keras.layers.BatchNormalization(name=self.name("_BNorm"))(layer)
        layer = tf.keras.layers.ReLU(self.relu_max, name=self.name("_ReLU"))(layer)
        return layer
