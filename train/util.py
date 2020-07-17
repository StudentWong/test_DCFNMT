import numpy as np
import cv2


def iou_cal(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def iou_xywh(bb_1, bb_2):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_test = bb_1.copy()
    bb_gt = bb_2.copy()
    if len(bb_test.shape) == 2:
        bb_test[:, 2:4] = bb_test[:, 2:4] + bb_test[:, 0:2]
        bb_gt[:, 2:4] = bb_gt[:, 2:4] + bb_gt[:, 0:2]
    else:
        bb_test[2:4] = bb_test[2:4] + bb_test[0:2]
        bb_gt[2:4] = bb_gt[2:4] + bb_gt[0:2]

    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def cxy_wh_2_rect1(pos, sz):
    return np.array([pos[0] - sz[0] / 2 + 1, pos[1] - sz[1] / 2 + 1, sz[0], sz[1]])  # 1-index


def rect1_2_cxy_wh(rect):
    return np.array([rect[0] + rect[2] / 2 - 1, rect[1] + rect[3] / 2 - 1]), np.array([rect[2], rect[3]])  # 0-index


def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])  # 0-index


# def gaussian_shaped_labels(sigma, sz):
#     x, y = np.meshgrid(np.arange(1, sz[1] + 1) - np.floor(float(sz[1]) / 2),
#                        np.arange(1, sz[0] + 1) - np.floor(float(sz[0]) / 2))
#     d = x ** 2 + y ** 2
#     g = np.exp(-0.5 / (sigma ** 2) * d)
#     g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
#     # g = np.roll(g, -1, axis=0)
#     g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
#     return g


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(0, sz[0]+0) - np.floor(float(sz[0]) / 2),
                       np.arange(0, sz[1]+0) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 0), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 0), axis=1)
    return g.astype(np.float32)


def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


def crop_chw2(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz[1] - 1) / (bbox[2] - bbox[0])
    b = (out_sz[0] - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz[1], out_sz[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return [np.transpose(crop, (2, 0, 1)), a, b]


def crop_chw_fix_scale(image, bbox, a, b, padding=(0, 0, 0)):
    w = int(b * (bbox[3] - bbox[1]) + 1)
    h = int(a * (bbox[2] - bbox[0]) + 1)
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (h, w), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


if __name__ == '__main__':
    print(iou_cal([50, 50, 100, 100], [75, 75, 125, 125]))
    g = gaussian_shaped_labels(10., [6, 5])
    print(g)
