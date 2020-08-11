import tensorflow as tf
import tflitegpu as tf_gpu


class MobileNetV1:
    def __init__(self, input_shape=(None, None, 3), n_classes=0, alpha=1.0):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.alpha = alpha
        self.initial_filter = int(32 * self.alpha)
        self.filters = [[32, 64], [64, 128], [128, 128], [128, 128], [256, 256], [256, 512]] + [[512, 512]]*5 + [[512, 1024], [1024, 1024]]
        self.reduces = [False, True, False, True, False, True] + [False]*5 +[True, False]

        for i in range(len(self.filters)):
            self.filters[i] = [int(self.filters[i][0] * self.alpha), int(self.filters[i][1] * self.alpha)]

    def get_layer(self, input_layer):
        layer = input_layer

        layer = tf_gpu.SeparableBatchConv2D(self.initial_filter, (3, 3),
                                            strides=(2, 2), padding='same', use_bias=False, prefix="l01")(layer)

        for i, x in enumerate(zip(self.filters, self.reduces)):
            n_filter, reduce = x

            layer = tf_gpu.MobileNetV1Block(n_filter, reduce_size=reduce, prefix=f"l{i:02d}")(layer)

        if self.n_classes > 0:
            layer = tf_gpu.GlobalAveragePooling2D()(layer)
            layer = tf.keras.layers.Dense(self.n_classes, activation='softmax')(layer)

        return layer

    def build_model(self):
        input_layer = tf.keras.layers.Input(self.input_shape)

        layer = self.get_layer(input_layer)

        model = tf.keras.models.Model(input_layer, layer)

        # model.summary()

        return model


# class MoibleNetV2:
#     def __init__(self, input_shape=(None, None, 3), n_classes=0, alpha=1.0):
#         self.input_shape = input_shape
#         self.n_classes = n_classes
#         self.alpha = alpha