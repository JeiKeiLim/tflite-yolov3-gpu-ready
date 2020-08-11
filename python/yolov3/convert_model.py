import tensorflow as tf
from util import prettyjson
from yolov3 import YoloJK


def convert_to_tflite_model(model, config):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    optimizations = []
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    supported_types = []

    if config['quantization']:
        optimizations += [tf.lite.Optimize.DEFAULT]
        supported_types += [config['quantization_type']]

        if config['quantization_type'] == tf.uint8:
            supported_ops += tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

    if config['tf_ops']:
        supported_ops += [tf.lite.OpsSet.SELECT_TF_OPS]

    converter.optimizations = optimizations
    converter.target_spec.supported_ops = supported_ops
    converter.target_spec.supported_types = supported_types

    converter.experimental_new_converter = config['exp_converter']

    tflite_model = converter.convert()

    with open(config['out_path'], 'wb') as f:
        f.write(tflite_model)

    print('Saved TFLite model to:', config['out_path'])

    return tflite_model


def parse_config(config):

    qtype = config['quantization_type']
    config['quantization_type'] = tf.float16 if qtype == "float16" else tf.int8 if qtype == "int8" else tf.float32

    print("\n{} TFLite Config {}".format("=" * 20, "=" * 20))
    print(prettyjson(config))
    print("{}".format("=" * 50))

    return config


def make_tflite(model_config, tflite_config):
    assert model_config is not None and tflite_config is not None

    tflite_config = parse_config(tflite_config)

    model_config['input_shape'] = tflite_config['input_shape']
    yolojk = YoloJK(model_config)
    yolojk.load_weights(load_train=False)

    convert_to_tflite_model(yolojk.model, tflite_config)
    tflite_interpreter = tf.lite.Interpreter(model_path=tflite_config['out_path'])

    print(tflite_interpreter.get_input_details())
    print(tflite_interpreter.get_output_details())

    return tflite_interpreter
