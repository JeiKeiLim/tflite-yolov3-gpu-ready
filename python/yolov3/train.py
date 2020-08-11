from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tfhelper.gpu import allow_gpu_memory_growth
import tensorflow as tf
import time
from tensorflow.python.framework import ops
import gc
from util import YoloLRScheduler, UnfreezeScheduler
from util import prettyjson
from . import YoloJK, YoloDecoder
import dataset


def test_predict_plot(model_, p_model, val_generator, test_generator, yolo_decoder, n_plots=1, figscale=20.0, batch_size_=16, eval=True, eval_steps=100, title_prefix=""):
    if n_plots < 0:
        n_plots = float('inf')

    if eval:
        print("... Evaluation")
        eval_loss = model_.evaluate(val_generator, steps=eval_steps)
        print("... Evaluation Loss: {}".format(eval_loss))
    else:
        eval_loss = None

    if n_plots == 0:
        return eval_loss

    a, b, c = next(test_generator)

    imgs, g_truths = a[0], a[1:]

    n_truths = [[np.expand_dims(g_truths[0][i].reshape((g_truths[0].shape[1], g_truths[0].shape[2], g_truths[0].shape[3] * g_truths[0].shape[4])), axis=0),
                 np.expand_dims(g_truths[1][i].reshape((g_truths[1].shape[1], g_truths[1].shape[2], g_truths[1].shape[3] * g_truths[1].shape[4])), axis=0),
                 np.expand_dims(g_truths[2][i].reshape((g_truths[2].shape[1], g_truths[2].shape[2], g_truths[2].shape[3] * g_truths[2].shape[4])), axis=0),
                 ] for i in range(batch_size_)]

    true_box_colors = [(0.3, 0.1, 0.8, 0.6),
                      (0.1, 0.3, 0.8, 0.6),
                      (0.1, 0.1, 0.8, 0.6),
                      (0.3, 0.3, 0.8, 0.6)]
    test_box_colors = [
        (color[2],) + color[0:2] + (color[-1],)
        for color in true_box_colors
    ]

    plot_num = 0
    for out_idx in range(batch_size_):
        boxes, scores, classes = p_model(np.expand_dims(imgs[out_idx], axis=0))

        if scores.shape[0] > 0:
            t_boxes, t_scores, t_classes_idx = yolo_decoder.decode_from_ground_truth(n_truths[out_idx], imgs[out_idx].shape[:2][::-1])
            fig, ax = yolo_decoder.plot_result(t_boxes, t_scores, t_classes_idx, box_colors=true_box_colors, img=imgs[out_idx], figscale=figscale, show=False)

            title_msg = f"{title_prefix}"
            title_msg += f"Eval loss: {eval_loss}" if eval else ""

            boxes, scores, classes = boxes.numpy(), scores.numpy(), classes.numpy()

            boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
            yolo_decoder.plot_result(boxes, scores, classes, img_shape=imgs[out_idx].shape,
                                     box_colors=test_box_colors, ax=ax, title=title_msg)
            fig.show()

            plot_num += 1

            if plot_num >= n_plots:
                break

    plt.close('all')
    return eval_loss


def train_model_bdd(model_, prediction_model, bdd, model_config, n_plots=2):
    train_config = model_config['train']

    unfreeze_by_step = train_config['unfreeze_by_step']
    unfreeze_idx = train_config['unfreeze_idx_from']
    steps_per_epoch = train_config['steps_per_epoch']
    validation_step = train_config['validation_step']
    val_batch_size = train_config['val_batch_size']
    save_path = train_config['model_save_path']
    learning_rate = train_config['learning_rate']
    batch_size = train_config['batch_size']
    last_lr = train_config['learning_rate_end']
    last_step_epoch = train_config['last_step_epoch']

    steps_per_epoch = len(bdd.train_label) // train_config['batch_size'] if steps_per_epoch == 0 else steps_per_epoch

    one_step_epoch = len(bdd.train_label) / (train_config['batch_size'] * steps_per_epoch)
    total_step_epochs = int(one_step_epoch * train_config['epochs'])

    tf_callbacks = [YoloLRScheduler(initial_lr=learning_rate, last_lr=last_lr,
                                    start_step_epoch=last_step_epoch, batch_size=batch_size,
                                    steps_per_epoch=steps_per_epoch,
                                    n_data=len(bdd.train_label))]
    if train_config['unfreeze_by_step'] > 0:
        tf_callbacks += [UnfreezeScheduler(epochs=last_step_epoch,
                                          wait_epoch=train_config['unfreeze_wait_step'],
                                          unfreeze_step=unfreeze_by_step,
                                          unfreeze_from=unfreeze_idx)]

    run_time = None
    run_times = []

    input_shape = bdd.config['input_shape']

    train_gen = bdd.data_generator(batch_size, input_shape, type="train", shuffle=True, augment=True)
    val_gen = bdd.data_generator(val_batch_size, input_shape, type="val", shuffle=False, augment=False)
    test_gen = bdd.data_generator(val_batch_size, input_shape, type="val", shuffle=False, augment=True)
    yolo_decoder = YoloDecoder(model_config)

    test_predict_plot(model_, prediction_model, val_gen, test_gen, yolo_decoder,
                      eval=True,
                      eval_steps=1,
                      n_plots=n_plots,
                      batch_size_=val_batch_size,
                      title_prefix="Step Epoch: {:03d} :: ".format(last_step_epoch))

    for i in range(last_step_epoch, total_step_epochs):
        actual_epoch = int(((train_config['batch_size']*steps_per_epoch) / len(bdd.train_label)) * i)

        start_time = time.time()

        n_trainable_layer = np.sum([model_.layers[idx].trainable for idx in range(len(model_.layers))])
        if run_time is None:
            run_time_str = "None"
        else:
            run_times.append(run_time)
            mean_run_time = np.mean(run_times)
            eta_time = mean_run_time*(total_step_epochs-i)
            run_time_str = "{:02d}:{:02d}:{:02d}".format(
                int(eta_time/60/60),
                int(eta_time/60) % 60,
                int(eta_time % 60))
            run_times = run_times[:10]

        lr = model_.optimizer.lr.numpy()

        print("... Training with {:03d}/{:03d} layers (trainable/non-trainable) ... LR: {:.8f}\n".format(n_trainable_layer, len(
            model_.layers) - n_trainable_layer, lr) +
              "... Actual Epoch: {:02d}/{:02d}, ".format(actual_epoch, train_config['epochs']) +
              "Step Epoch: {:03d}/{:03d} ({:.2f}% Complete. Estimated Time: {})".
              format(i + 1, total_step_epochs, (i / total_step_epochs) * 100, run_time_str
                     ))

        model_.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=1, callbacks=tf_callbacks)

        loss = test_predict_plot(model_, prediction_model,
                                 val_gen, test_gen, yolo_decoder,
                                 eval=True,
                                 eval_steps=validation_step,
                                 n_plots=n_plots,
                                 batch_size_=val_batch_size,
                                 title_prefix="Step Epoch: {:03d} :: ".format(i+1))
        run_time = time.time() - start_time

        if (loss >= 0 or loss <= 0) and save_path != "":
            model_.save_weights(save_path)
        elif not (loss >= 0 or loss <= 0):
            print("Loss is nan. Stop training.")
            return

        tf.keras.backend.clear_session()
        ops.reset_default_graph()
        gc.collect()

        train_config['last_step_epoch'] = i
        with open(model_config['self_path'], 'w') as f:
            f.write(prettyjson(model_config, maxlinelength=20))


def train_yolo(dataset_config, model_config, n_plot=2):
    assert dataset_config is not None and model_config is not None

    bdd100k = dataset.BDD100kDataset(dataset_config, load_anchor=True)

    allow_gpu_memory_growth()

    train_config = model_config['train']

    input_shape = dataset_config['input_shape']
    learning_rate = train_config['learning_rate']

    # Plot example images START
    for idx in np.random.randint(0, len(bdd100k.val_label), n_plot):
        bdd100k.plot_label(idx, prefix="val")

    for idx in np.random.randint(0, len(bdd100k.val_label), n_plot):
        img, box = bdd100k.get_preprocessed_data(idx, input_shape, type="val", augment=True)
        bdd100k.plot_bboxes(img, box)
    # Plot example images END

    yolojk = YoloJK(model_config)
    yolojk.load_weights(load_train=True)
    model = yolojk.build_loss_model()
    p_model = yolojk.build_prediction_model()

    model.compile(optimizer=Adam(lr=learning_rate), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    train_model_bdd(model, p_model, bdd100k, model_config, n_plots=n_plot)
