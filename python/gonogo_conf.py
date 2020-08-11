import json
import argparse


def get_arg_parser(return_config=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--n-plot", default=2, type=int)
    parser.add_argument("--dataset-config", default="./conf/bdd100k.json", type=str)
    parser.add_argument("--model-config", default="./conf/yolo_conf.json", type=str)
    parser.add_argument("--tflite-config", default="./conf/tflite_conf.json", type=str)
    parser.add_argument("--video", default="", type=str)
    parser.add_argument("--video-w", default=416, type=int)
    parser.add_argument("--video-h", default=256, type=int)
    parser.add_argument("--video-rot", default=0, type=int)
    parser.add_argument("--video-show", dest='video_show', action='store_true', default=True)
    parser.add_argument("--no-video-show", dest='video_show', action='store_false')
    parser.add_argument("--video-out", default="", type=str)
    parser.add_argument("--video-fourcc", default="mp4v", type=str)
    parser.add_argument("--verbose", default=1, type=int)

    args = parser.parse_args()

    d_conf = get_config(args.dataset_config) if return_config else None
    m_conf = get_config(args.model_config) if return_config else None
    tf_conf = get_config(args.tflite_config) if return_config else None

    return parser, d_conf, m_conf, tf_conf


def get_config(path):
    try:
        with open(path, 'r') as f:
            conf = json.load(f)

        conf['self_path'] = path
    except IOError:
        print(f"Error Open {path}")
        return None

    return conf
