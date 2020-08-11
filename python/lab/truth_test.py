from gonogo_conf import get_arg_parser
from yolov3 import YoloDecoder
from dataset.bdd100k_dataset import BDD100kDataset
import numpy as np

if __name__ == '__main__':
    parser, dataset_config, model_config, tflite_config = get_arg_parser(return_config=True)
    args = parser.parse_args()

    bdd100k = BDD100kDataset(dataset_config, load_anchor=True)
    input_shape = dataset_config['input_shape']

    yolo_decoder = YoloDecoder(model_config)

    d_gen = bdd100k.data_generator(16, dataset_config['input_shape'], type="val", shuffle=False)
    a, b, c = next(d_gen)

    imgs, g_truths = a[0], a[1:]

    n_truths = [[np.expand_dims(g_truths[0][i].reshape((g_truths[0].shape[1], g_truths[0].shape[2], g_truths[0].shape[3] * g_truths[0].shape[4])), axis=0),
                 np.expand_dims(g_truths[1][i].reshape((g_truths[1].shape[1], g_truths[1].shape[2], g_truths[1].shape[3] * g_truths[1].shape[4])), axis=0),
                 np.expand_dims(g_truths[2][i].reshape((g_truths[2].shape[1], g_truths[2].shape[2], g_truths[2].shape[3] * g_truths[2].shape[4])), axis=0),
                 ] for i in range(16)]

    # img_idx = 5
    true_box_colors = [(0.3, 0.1, 0.8, 0.6),
                      (0.1, 0.3, 0.8, 0.6),
                      (0.1, 0.1, 0.8, 0.6),
                      (0.3, 0.3, 0.8, 0.6)]
    test_box_colors = [
        (color[2],) + color[0:2] + (color[-1],)
        for color in true_box_colors
    ]

    for img_idx in range(16):
        true_boxes = c[img_idx, :, :4]
        mask = true_boxes.sum(axis=1) > 0
        true_boxes = true_boxes[mask]
        true_classes = c[img_idx, mask, 4].astype(np.int)
        true_confs = np.ones(true_classes.shape[0], dtype=np.float)

        t_boxes, t_scores, t_classes = yolo_decoder.decode_from_ground_truth(n_truths[img_idx], imgs.shape[1:3][::-1])
        fig, ax = yolo_decoder.plot_result(true_boxes, true_confs, true_classes, img=imgs[img_idx], figscale=20, box_colors=true_box_colors, show=False)
        yolo_decoder.plot_result(t_boxes, t_scores, t_classes, ax=ax, img_shape=imgs[img_idx].shape,
                                 box_colors=test_box_colors, title=f"Ground Truth Check : {img_idx:02d}")
        fig.show()
