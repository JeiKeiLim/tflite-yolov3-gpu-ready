import cv2
import numpy as np
import time
from tqdm import tqdm
from yolov3 import YoloJK, YoloLoss


def test_video(video_path, model_config, video_size=(416, 256), video_rotation=0,
               save_video_path="", show_video=True, fourcc="MP4V"):
    """

    :param video_path:
    :param dataset_config:
    :param model_config:
    :param video_size:
    :param video_rotation: 0: No rotation, 1: ROTATE_90_COUNTERCLOCKWISE, 2: ROTATE_90_CLOCKWISE
    :param save_video_path:
    :param show_video:
    :param fourcc:
    :return:
    """
    assert model_config is not None

    if len(video_path) < 3:
        try:
            video_path = int(video_path)
        except:
            print("Wrong format!")
            return

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    if not cap.isOpened():
        print("Error opening video  file")

    anchors = np.array(model_config['anchors'])
    use_classes = model_config['class_ids']
    n_classes = len(use_classes)
    input_shape = (video_size[1], video_size[0])

    model_config['input_shape'] = [input_shape[0], input_shape[1], 3]
    yolojk = YoloJK(model_config)
    model = yolojk.build_model()
    model = YoloLoss(model_config, model).loss_model

    model.load_weights(model_config['weight_path'])

    p_model = yolojk.build_prediction_model(input_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5)

    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_rotation > 0:
        v_width, v_height = v_height, v_width

    v_fps = cap.get(cv2.CAP_PROP_FPS)

    print(" Video Resolution: {}x{}, FPS: {:.1f}, Time: {:.1f}s".format(v_width, v_height, v_fps, max_frame/v_fps))
    print("Target Resolution: {}x{}".format(video_size[0], video_size[1]))

    if save_video_path != "":
        v_out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*fourcc),
                                v_fps, (video_size[0], video_size[1]))
    else:
        v_out = None

    p_bar = tqdm(total=max_frame, desc="YOLO")
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        # Display the resulting frame
        if video_rotation == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif video_rotation == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame = cv2.resize(frame, video_size, frame)

        yolo_frame = np.expand_dims(frame, axis=0) / 255
        m_time_s = time.time()

        boxes, scores, classes = p_model.predict(yolo_frame)
        m_time = time.time() - m_time_s

        end_time = time.time()
        fps = 1/(end_time-start_time)
        fps_msg = "FPS: ({:.1f}, {:.1f})".format(1/m_time, fps)

        # cv2.putText(frame, fps_msg, (int(frame.shape[0] * 0.05), int(frame.shape[1] * 0.5)),
        #             cv2.FONT_HERSHEY_SIMPLEX, np.log(frame.shape[0] * frame.shape[1]) / 20, (255, 0, 0))
        cv2.putText(frame, fps_msg, (int(frame.shape[0] * 0.05), int(frame.shape[1] * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, np.log(frame.shape[0] * frame.shape[1]) / 20, (255, 0, 0))

        if scores.shape[0] > 0:
            for box_idx, box in enumerate(boxes):
                x, y, x2, y2 = box[1], box[0], box[3], box[2]
                msg = "{} ({:.2f}%)".format(use_classes[classes[box_idx]], scores[box_idx]*100)

                frame = cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, msg, (int(x + ((x2-x)/2)), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))

        if v_out:
            v_out.write(frame)

        if show_video:
            frame = cv2.resize(frame, (v_width, v_height), frame)

            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        p_bar.set_description("YOLO {}".format(fps_msg))
        p_bar.update()

    if v_out:
        v_out.release()

    p_bar.close()