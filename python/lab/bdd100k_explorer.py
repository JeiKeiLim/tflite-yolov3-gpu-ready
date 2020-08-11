from yolov3 import train_yolo
import dataset
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


if __name__ == "__main__":
    with open("./conf/bdd100k.json") as f:
        d_conf = json.load(f)

    bdd100k = dataset.BDD100kDataset(d_conf)
    img, labels = bdd100k.get_sample_data(13881)
    n_img = np.array(img)

    fig, ax = plt.subplots()

    for i in range(len(labels)):
        if labels[i]['category'] == "traffic light":
            x1 = labels[i]['box2d']['x1']
            x2 = labels[i]['box2d']['x2']
            y1 = labels[i]['box2d']['y1']
            y2 = labels[i]['box2d']['y2']

            i_x1 = np.round(x1).astype(np.int)
            i_x2 = np.round(x2).astype(np.int)
            i_y1 = np.round(y1).astype(np.int)
            i_y2 = np.round(y2).astype(np.int)

            w = x2 - x1
            h = y2 - y1
            print(h / w)

            i_w = i_x2 - i_x1
            i_h = i_y2 - i_y1

            c_img = n_img[i_y1:i_y2, i_x1:i_x2, :]
            r_img = np.swapaxes(c_img, 0, 1).astype(np.uint8)

            m_img = []
            for iy in range(max(i_y1-i_h-5, 0), min(i_y1+i_h+5, n_img.shape[0]), i_h):
                for ix in range(max(i_x1-i_w-5, 0), min(i_x1+i_w+5, n_img.shape[1]), i_w):
                    m_img.append(n_img[iy:iy+i_h, ix:ix+i_w, :])

            m_img = np.array(m_img).mean(axis=0).astype(np.uint8)
            n_img[i_y1:i_y2, i_x1:i_x2, :] = m_img

            n_img[i_y1+(i_w//2):(i_y1+i_w)+(i_w//2), i_x1-(i_h//2):(i_x1+i_h)-(i_h//2)] = r_img

            n_y1 = y1 + (w*0.5)
            n_y2 = y1 + (w*1.5)
            n_x1 = x1 - (h*0.5)
            n_x2 = (x1+h) - (h*0.5)

            rect = patches.Rectangle((n_x1, n_y1), n_x2-n_x1, n_y2-n_y1,
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            # n_img[i_y1:(i_y1 + i_w), i_x1:(i_x1 + i_h)] = r_img
    plt.figure(figsize=(20, 15))
    plt.imshow(n_img)
    plt.show()

    for idx in np.random.randint(0, len(bdd100k.train_label), 10):
        bdd100k.plot_label(idx, prefix="train", show_attr=True, show_bbox_ratio=True, title="idx: {}".format(idx))

    attrs = [list(label['attributes'].values()) + list(label['attributes'].keys())
             for label in labels for labels in bdd100k.train_label]
    unique_attrs = np.unique(attrs)
