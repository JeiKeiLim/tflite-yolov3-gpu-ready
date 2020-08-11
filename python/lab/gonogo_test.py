import cv2
import numpy as np
import time
from tqdm import tqdm
from yolov3 import YoloJK, YoloLoss
import time


class TimeMeasure:
    def __init__(self, verbose=0, prefix=""):
        self.verbose = verbose
        self.prefix = prefix
        self.start_time = None
        self.results = {}

    def start(self):
        self.start_time = time.time()

        return self

    def end(self, prefix=""):
        if self.start_time is None:
            return

        if prefix == "":
            prefix_msg = self.prefix
        else:
            prefix_msg = prefix

        run_time = time.time() - self.start_time
        if self.verbose > 0:
            print(f"[{prefix_msg}] took {run_time:.3f}s, FPS: {1/run_time:.1f}")

        self.start_time = None

        i = 0
        while True:
            if prefix_msg not in self.results.keys():
                break
            else:
                prefix_msg = prefix_msg[-2:] if i > 0 else prefix_msg
                prefix_msg = f"{prefix_msg}{i:02d}"
                i += 1

        self.results[prefix_msg] = run_time
        return run_time

    def get_total_time(self):
        return np.sum(list(self.results.values()))

    def print_results(self):
        sum_time = self.get_total_time()

        for i, key in enumerate(self.results.keys()):
            print(f"{i:02d} :: [{key}] took {self.results[key]:.3f}s, FPS: {1/self.results[key]:.1f}, {(self.results[key]/sum_time)*100:.2f}%")

        print(f"Total Time took {sum_time:.3f}s, FPS: {1/sum_time}")

    def clear(self):
        self.results = {}

    def __call__(self, fn, *args, prefix="", **kwargs, ):
        self.start()
        result = fn(*args, **kwargs)

        prefix_msg = self.prefix if prefix == "" else prefix

        run_time = self.end(prefix=prefix_msg)

        if self.verbose > 0 or self.prefix == "":
            return result
        else:
            return result, run_time


def test_video_gonogo(video_path, p_model, class_ids, delegate, video_size=(416, 256), video_rotation=0,
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
        return

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

    p_bar = tqdm(total=max_frame, desc="YOLO", disable=True)
    while cap.isOpened():
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Display the resulting frame
        if video_rotation == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif video_rotation == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame = cv2.resize(frame, video_size, frame)

        yolo_frame = np.expand_dims(frame, axis=0) / 255
        boxes, scores, classes = p_model.predict(yolo_frame)

        delegate(frame, boxes, scores, classes)

        end_time = time.time()
        fps = 1/(end_time-start_time)
        fps_msg = "{:.1f} FPS".format(fps)

        cv2.putText(frame, fps_msg, (int(frame.shape[0] * 0.05), int(frame.shape[1] * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, np.log(frame.shape[0] * frame.shape[1]) / 20, (255, 0, 0))

        if scores.shape[0] > 0:
            for box_idx, box in enumerate(boxes):
                x, y, x2, y2 = box[1], box[0], box[3], box[2]
                msg = "{} ({:.2f}%)".format(class_ids[classes[box_idx]], scores[box_idx]*100)

                frame = cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, msg, (int(x + ((x2-x)/2)), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))

        if v_out:
            v_out.write(frame)

        if show_video:
            frame = cv2.resize(frame, (v_width, v_height), frame)

            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            key_in = cv2.waitKey(25) & 0xFF
            quit_options = np.array([ord('q'), ord('n'), ord('p')])
            if key_in in quit_options:
                if v_out:
                    v_out.release()
                p_bar.close()

                return key_in

        p_bar.set_description("YOLO {}".format(fps_msg))
        p_bar.update()

    if v_out:
        v_out.release()

    return ord('!')


class GoNoGo:
    def __init__(self, model_config, video_size):
        self.config = model_config
        self.video_size = video_size
        self.canvas = np.ones(video_size, np.uint8) * 255

        self.last_frame = None
        self.diff_frame = np.ones((10, 10, 3), np.uint8)
        self.mean_frame = np.ones((10, 10, 3), np.uint8)

        self.diff_count = 0
        self.diff_delay_frame = 15
        self.diff_value = np.inf
        self.diff_threshold = 5

        self.gonogo = []
        self.box_sizes = []
        self.window_size = 30
        self.gonogo_value = 0

        self.traffic_rgb = [[0, 0, 0]]

        self.state = {'car_stop': False, 'should_go': False}
        self.time_checker = TimeMeasure(verbose=0)

    def init_values(self):
        self.last_frame = None
        self.diff_frame = np.ones((10, 10, 3), np.uint8)
        self.mean_frame = np.ones((10, 10, 3), np.uint8)

        self.diff_count = 0
        self.diff_delay_frame = 10
        self.diff_value = np.inf
        self.diff_threshold = 5

        self.gonogo = []
        self.box_sizes = []
        self.window_size = 60
        self.gonogo_value = 0

        self.state['car_stop'] = False
        self.state['should_go'] = False
        self.traffic_rgb = [[0, 0, 0]]

    def norm_frame(self, frame, min_sat=10, max_sat=90):
        m_frame = np.array(frame, dtype=np.float)
        min_per, max_per = np.percentile(m_frame, [min_sat, max_sat], axis=(0, 1))

        m_frame -= min_per
        m_frame /= (max_per-min_per)
        m_frame[m_frame < 0] = 0
        m_frame[m_frame > 1] = 1

        m_frame = np.array(m_frame * 255, dtype=np.uint8)

        tmp_frame = cv2.resize(m_frame, (m_frame.shape[1], m_frame.shape[0]))
        CVText(fs=0.5, color=(0, 0, 255), xr=0.05, yr=0.1, thickness=2)(tmp_frame, f"MIN: {min_per}, MAX: {max_per}")
        cv2.imshow('norm_frame', tmp_frame)

        return m_frame

    def mean_conv(self, frame):
        m_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        m_frame = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)

        m_frame = self.time_checker(self.norm_frame, m_frame, prefix="!! norm_frame !!")

        w_size = np.array([int(m_frame.shape[0] / 5), int(m_frame.shape[1] / 5)])
        stride = (w_size / 1.5).astype(np.int)

        if len(m_frame.shape) == 3:
            m_conv_fn = lambda x, si, st: np.array([[np.median(x[i:i + si[0], j:j + si[1], :], axis=(0, 1))
                              for j in range(0, x.shape[1], st[1])] for i in range(0, x.shape[0], st[0])])
        else:
            m_conv_fn = lambda x, si, st: np.array([[np.median(x[i:i + si[0], j:j + si[1]])
                                                     for j in range(0, x.shape[1], st[1])] for i in range(0, x.shape[0], st[0])])

        m_frame = self.time_checker(m_conv_fn, m_frame, w_size, stride, prefix="!! m_conv_fn !!")

        if len(m_frame.shape) == 2:
            m_frame = np.tile(np.expand_dims(m_frame, axis=-1), (1, 1, 3))

        return m_frame

    def stop_go_check(self, p_frame, frame, boxes, scores, classes):
        self.diff_count = (self.diff_count + 1) % self.diff_delay_frame
        self.canvas[-self.diff_frame.shape[0]:, -self.diff_frame.shape[1]:, :] = self.diff_frame
        self.canvas[-self.mean_frame.shape[0]:,-(self.diff_frame.shape[1] + self.mean_frame.shape[1]):-(self.diff_frame.shape[1])] = self.mean_frame

        frame[-self.diff_frame.shape[0]:, -self.diff_frame.shape[1]:, :] = self.diff_frame
        frame[-self.mean_frame.shape[0]:, -(self.diff_frame.shape[1] + self.mean_frame.shape[1]):-(self.diff_frame.shape[1])] = self.mean_frame

        gonogo = True if self.diff_value > self.diff_threshold else False
        self.gonogo.append(gonogo)
        self.gonogo = self.gonogo[-self.window_size:]
        self.gonogo_value = np.mean(self.gonogo)
        self.state['car_stop'] = True if self.gonogo_value < 0.3 else False

        CVText(fs=0.3, color=(0, 0, 255), xr=0.05, yr=0.95)(frame, f"Diff: {self.diff_value:.1f}, {'NOGO' if self.state['car_stop'] else 'GO'} : {self.gonogo_value:.1f}")

        if self.diff_count == 0:
            mean_frame = self.mean_conv(p_frame)

            if self.last_frame is not None:
                self.diff_frame = mean_frame - self.last_frame
                self.diff_value = np.abs(self.diff_frame).sum() / self.diff_frame.size
                self.diff_frame = cv2.resize(self.diff_frame,
                                             (self.diff_frame.shape[1] * 2, self.diff_frame.shape[0] * 2))

            self.last_frame = mean_frame
            self.mean_frame = cv2.resize(mean_frame,  (mean_frame.shape[1]*2, mean_frame.shape[0]*2))

    def draw_boxed_objects(self, frame, boxes, scores, classes):
        xs = 0
        ys = 0

        max_h = 0
        box_frames = []
        for i, box in enumerate(boxes):
            y1, x1, y2, x2 = box.astype(np.int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

            h, w = (y2 - y1), (x2 - x1)

            if h <= 1 or w <= 1:
                continue

            max_h = max(max_h, h)

            if xs + w + 1 >= self.canvas.shape[1]:
                xs = 0
                ys += max_h
                max_h = 0
                self.canvas[ys - 1, :, :] = [0, 0, 255]

            h = h - max(0, (ys+h)-self.canvas.shape[0])
            w = w - max(0, (xs+w)-self.canvas.shape[1])

            box_frame = frame[y1:y1 + h, x1:x1 + w, :]
            box_frames.append(box_frame)
            w = int(w * 1.0)
            h = int(h * 1.0)
            box_frame = cv2.resize(box_frame, (w, h))

            self.canvas[ys:ys + h, xs:xs + w, :] = box_frame
            self.canvas[ys:ys + h, xs + w, :] = [0, 0, 255]
            xs += w + 1

        return box_frames

    def video_in(self, frame, boxes, scores, classes):
        self.canvas[...] = 255
        p_frame = np.array(frame)
        self.time_checker(self.stop_go_check, p_frame, frame, boxes, scores, classes, prefix="## stop_go_check ##")
        # self.stop_go_check(p_frame, frame, boxes, scores, classes)

        box_frames = self.time_checker(self.draw_boxed_objects, p_frame, boxes, scores, classes, prefix="@@ draw_boxed_objects @@")
        # box_frames = self.draw_boxed_objects(p_frame, boxes, scores, classes)
        self.time_checker.start()

        x_center = frame.shape[1] / 2
        r_sum = 0
        g_sum = 0
        b_sum = 0
        max_box_size = 0
        for box, bframe, class_id in zip(boxes, box_frames, classes):
            if class_id == 6:
                r_sum += bframe[..., 2].sum() / len(box_frames) / bframe[..., 2].size / 255
                g_sum += bframe[..., 1].sum() / len(box_frames) / bframe[..., 1].size / 255
                b_sum += bframe[..., 0].sum() / len(box_frames) / bframe[..., 0].size / 255
            elif class_id in [0, 1, 2]:
                box_center_ratio = 1 - (np.abs(box[[1, 3]].mean() - x_center) / x_center)
                box_size = np.sqrt(bframe[..., 0].size) * box_center_ratio
                max_box_size = max(max_box_size, box_size)

        box_go = False
        traffic_go = False
        traffic_score = 0
        tl_rgb = np.median(self.traffic_rgb, axis=0)
        if self.gonogo_value < 0.5:
            if max_box_size > (frame.shape[1]/10):
                self.box_sizes.append(max_box_size)
                self.box_sizes = self.box_sizes[-(self.window_size*4):]

                if max_box_size < (np.median(self.box_sizes) * 0.8) and len(self.box_sizes) > (self.window_size):
                    box_go = True

            if (r_sum + g_sum + b_sum) > 0:
                self.traffic_rgb.append([r_sum, g_sum, b_sum])
                self.traffic_rgb = self.traffic_rgb[-(self.window_size*4):]

                traffic_score = (tl_rgb[0] - r_sum) + (g_sum - tl_rgb[1])
                if traffic_score > 0.1:
                    traffic_go = True
        elif self.gonogo_value > 0.9:
            self.box_sizes = []
            self.traffic_rgb = [[0, 0, 0]]

        CVText(fs=0.3, color=(0, 255, 0), xr=0.05, yr=0.9)(frame, f"R: {tl_rgb[0]:.2f}, G: {tl_rgb[1]:.2f}, B: {tl_rgb[2]:.2f}, TL Score: {traffic_score:.3f}")
        CVText(fs=0.3, color=(255, 0, 0), xr=0.05, yr=0.85)(frame, f"Box Size: {max_box_size:.2f}, ({np.median(self.box_sizes):.2f})")

        if box_go or traffic_go:
            CVText(fs=2.0, color=(0, 255, 0), xr=0.25, yr=0.5, thickness=2)(frame, "GO!" if box_go else "GO!!!")

        cv2.imshow('gonogo', cv2.resize(self.canvas, (self.video_size[1], self.video_size[0])))

        self.time_checker.end(prefix="** rest_code **")

        if self.time_checker.get_total_time() > 0.01:
            self.time_checker.print_results()
            print("-" * 30 + "\n")

        self.time_checker.clear()


class CVText:
    def __init__(self, fs=1.0, xr=0.1, yr=0.1, loc=None, color=(0, 0, 255), bottomLeftOrigin=None, lineType=None, thickness=None, font=cv2.FONT_HERSHEY_SIMPLEX):
        assert (loc is not None and len(loc) == 2) or loc is None
        assert isinstance(thickness, type(None)) or isinstance(thickness, int)
        self.loc = loc
        self.xr = xr
        self.yr = yr
        self.fs = fs
        self.color = color
        self.font = font
        self.thickness = thickness
        self.bottomLeftOrigin = bottomLeftOrigin
        self.lineType = lineType

    def __call__(self, frame, msg):
        if self.loc is not None:
            x = int(self.loc[0])
            y = int(self.loc[1])
        else:
            x = int(frame.shape[1] * self.xr)
            y = int(frame.shape[0] * self.yr)

        kwargs = {}
        if self.thickness:
            kwargs['thickness'] = self.thickness
        if self.lineType:
            kwargs['lineType'] = self.lineType
        if self.bottomLeftOrigin:
            kwargs['bottomLeftOrigin'] = self.bottomLeftOrigin

        cv2.putText(frame, msg, (x, y), self.font, self.fs, self.color, **kwargs)


def test_gonogo(video_root, model_config, video_size=(416, 256), video_rotation=0,
                      save_video_path="", show_video=True, fourcc="MP4V"):
    video_list = [  ["test/cbf84011-2b35cf8b.mov", 1],
                    ["train/0a0c3694-4cc8b0e3.mov", 1],
                    ["train/0a0c3694-24b5193a.mov", 1],
                    ["train/0a0cc110-7f2fd761.mov", 1],
                    ["train/0a0ceca1-4148e482.mov", 1],
                    ["train/0a2c99b4-d68c2338.mov", 2],
                    ["train/0a3bb2d8-e2bd5aea.mov", 2],
                    ["train/0a3f8f94-2208fb12.mov", 2],
                    ["train/0a3f965a-0198d7ae.mov", 1],
                    ["train/0a05a47b-2342eda4.mov", 1] ]

    class_ids = model_config['class_ids']
    input_shape = (video_size[1], video_size[0])

    model_config['input_shape'] = [input_shape[0], input_shape[1], 3]
    yolojk = YoloJK(model_config)
    yolojk.load_weights()

    gonogo = GoNoGo(model_config, (video_size[1], video_size[0], 3))

    p_model = yolojk.build_prediction_model(input_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5)

    v_idx = 0
    while video_list:
        gonogo.init_values()
        v_path, v_rot = video_list[v_idx]

        video_path = f"{video_root}/{v_path}"

        video_name = v_path[-21:-4]
        type = v_path.split("/")[0]

        save_path = f"{save_video_path[:-4]}_{type}_{video_name}.mp4" if save_video_path != "" else ""
        print(f"Running {video_path}")

        stop = test_video_gonogo(video_path, p_model, class_ids, gonogo.video_in,
                                 video_size=video_size, video_rotation=v_rot,
                                 save_video_path=save_path, show_video=show_video,
                                 fourcc=fourcc)
        if stop == ord('q'):
            break
        elif stop == ord('n'):
            v_idx = (v_idx + 1) % len(video_list)
        elif stop == ord('p'):
            v_idx = (v_idx - 1) if v_idx > 0 else len(video_list)-1
        elif stop == ord('!'):
            video_list.pop(v_idx)

            if video_list:
                v_idx = (v_idx + 1) % len(video_list)

