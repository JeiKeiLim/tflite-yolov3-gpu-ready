from gonogo_conf import get_arg_parser
from yolov3 import train_yolo
from yolov3 import make_tflite
from dataset import BDD100kDataset
from yolov3 import test_video
from util import prettyjson
from lab.gonogo_test import test_gonogo

if __name__ == '__main__':
    parser, dataset_config, model_config, tflite_config = get_arg_parser(return_config=True)

    args = parser.parse_args()

    print("[{}] {} Running {}".format(args.mode, "=" * 20, "=" * 20))
    print("[{}] {} Dataset Config {}".format(args.mode, "=" * 20, "=" * 20))
    print(prettyjson(dataset_config))
    print("\n[{}] {} Training Config {}".format(args.mode, "=" * 20, "=" * 20))
    print(prettyjson(model_config))
    print("[{}] {}".format(args.mode, "=" * 50))

    if args.mode == "train":
        train_yolo(dataset_config, model_config, n_plot=args.n_plot)
    elif args.mode == "tflite":
        tflite_interpreter = make_tflite(model_config, tflite_config)
    elif args.mode == "kmeans":
        bdd100k = BDD100kDataset(dataset_config, load_anchor=False)
    elif args.mode == "video":
        test_video(args.video, model_config,
                   video_size=(args.video_w, args.video_h),
                   video_rotation=args.video_rot,
                   save_video_path=args.video_out,
                   show_video=args.video_show,
                   fourcc=args.video_fourcc
                   )
    elif args.mode == "experimental":
        test_gonogo(args.video, model_config,
                   video_size=(args.video_w, args.video_h),
                   video_rotation=args.video_rot,
                   save_video_path=args.video_out,
                   show_video=args.video_show,
                   fourcc=args.video_fourcc
                   )
    else:
        parser.print_help()
