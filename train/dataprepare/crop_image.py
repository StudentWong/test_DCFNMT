from os.path import join, isdir, abspath
from os import mkdir
import argparse
import numpy as np
import json
import cv2
import time

# print(np.random.normal(loc=0.0, scale=60, size=1))

parse = argparse.ArgumentParser(description='Generate training data (cropped) for DCFNet_pytorch')
parse.add_argument('-v', '--visual', dest='visual', action='store_true', help='whether visualise crop')
parse.add_argument('-o', '--output_size', dest='output_size', default=103, type=int, help='crop output size')
parse.add_argument('-p', '--padding', dest='padding', default=1, type=float, help='crop padding size')
parse.add_argument('-d', '--disturbance', dest='disturbance', default=0.3, type=float, help='disturbance decay')
parse.add_argument('-n', '--number', dest='num', default=2, type=int, help='disturbance number')

args = parse.parse_args()

print(args)


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    bbox = [float(x) for x in bbox]
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return [crop, a, b]


def crop_hwc_with_disturbance(image, bbox, distrubance, out_sz, padding=(0, 0, 0)):
    bbox = [float(x) for x in bbox]
    distrubance = [float(x) for x in distrubance]
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])  # 0-index


basepath = abspath(".")
snip_path = join(basepath, 'train', 'dataprepare', 'snippet.json')
snaps = json.load(open(snip_path, 'r'))

num_all_frame = 245098  # cat snippet.json | grep bbox |wc -l
num_val = 1000
# crop image
max_distrubance_bias = (args.padding / (1 + args.padding)) / 2
disturbance_factor = [0.1, 0.3, 0.5, 0.7]
T_lenth = 20

lmdb = dict()
lmdb['down_index'] = np.zeros(num_all_frame, np.int)  # buff
lmdb['up_index'] = np.zeros(num_all_frame, np.int)

crop_base_path = 'crop_{:d}_{:1.1f}_{:1.1f}'.format(args.output_size, args.padding, args.disturbance)
if not isdir(crop_base_path):
    mkdir(crop_base_path)

count = 0
begin_time = time.time()
for sn, snap in enumerate(snaps):
    snappath = join(crop_base_path, '{:08d}'.format(sn))
    if not isdir(snappath):
        mkdir(snappath)
    frames = snap['frame']
    n_frames = len(frames)
    for f, frame in enumerate(frames):
        img_path = join(snap['base_path'], frame['img_path'])
        im = cv2.imread(img_path)
        avg_chans = np.mean(im, axis=(0, 1))
        bbox = frame['obj']['bbox']

        target_pos = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
        target_sz = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])

        template_name = join(snappath,
                             'tp_f{:05d}.jpg'
                             .format(f))

        window_sz = target_sz * (1 + args.padding)
        crop_bbox = cxy_wh_2_bbox(target_pos, window_sz)
        [patch, a2, b2] = crop_hwc(im, crop_bbox, args.output_size)

        cv2.imwrite(template_name, patch)
        # print(template_name)


        for d_f in disturbance_factor:
            dscale = d_f * max_distrubance_bias * target_sz

            target_pos_distrubance_x = np.random.normal(loc=0.0, scale=dscale[0], size=args.num)
            target_pos_distrubance_y = np.random.normal(loc=0.0, scale=dscale[1], size=args.num)

            for ii in range(0, args.num):

                # print([target_pos_distrubance_x[ii], target_pos_distrubance_y[ii]])
                if target_pos_distrubance_x[ii] > max_distrubance_bias * window_sz[0]:
                    target_pos_distrubance_x[ii] = max_distrubance_bias * window_sz[0]
                if target_pos_distrubance_x[ii] < -max_distrubance_bias * window_sz[0]:
                    target_pos_distrubance_x[ii] = -max_distrubance_bias * window_sz[0]

                if target_pos_distrubance_y[ii] > max_distrubance_bias * window_sz[1]:
                    target_pos_distrubance_y[ii] = max_distrubance_bias * window_sz[1]
                if target_pos_distrubance_y[ii] < -max_distrubance_bias * window_sz[1]:
                    target_pos_distrubance_y[ii] = -max_distrubance_bias * window_sz[1]

                target_pos_with_distrubance_x = round(target_pos[0] + target_pos_distrubance_x[ii])
                target_pos_with_distrubance_y = round(target_pos[1] + target_pos_distrubance_y[ii])
                target_pos_with_distrubance = [target_pos_with_distrubance_x, target_pos_with_distrubance_y]

                crop_bbox_d = cxy_wh_2_bbox(target_pos_with_distrubance, window_sz)
                [patch_d, a1, b1] = crop_hwc(im, crop_bbox_d, args.output_size)



                a = (a1 + a2) / 2
                b = (b1 + b2) / 2
                target_pos_distrubance_x_small = round(a * target_pos_distrubance_x[ii])
                target_pos_distrubance_y_small = round(b * target_pos_distrubance_y[ii])

                # print(target_pos_distrubance_x_small)
                # print(target_pos_distrubance_y_small)
                # print([target_pos_distrubance_x[ii], target_pos_distrubance_y[ii]])
                # print(target_pos_with_distrubance)
                # cv2.imshow("0", patch)
                # cv2.waitKey(0)
                # cv2.imshow("0", patch_d)
                # cv2.waitKey(0)
                # cv2.imshow("0", patch_nopadding)
                # cv2.waitKey(0)
                disturbance_name = join(snappath,
                                        'f{:05d}_d{:1.1f}_n{:01d}_x{:d}_y{:d}.jpg'
                                        .format(f, d_f,
                                                ii,
                                                int(target_pos_distrubance_x_small),
                                                int(target_pos_distrubance_y_small)))

                # print(disturbance_name)
                cv2.imwrite(disturbance_name, patch_d)

                # cv2.imwrite(join(snappath, 'f{:08d}_d{:1.1f}_x{:02d}_y{:02d}.jpg'.format(count)), patch)
                # cv2.imwrite('crop.jpg'.format(count), patch)

        # lmdb['down_index'][count] = f
        # lmdb['up_index'][count] = n_frames - f
                count += 1
                if count % 100 == 0:
                    elapsed = time.time() - begin_time
                    print("Processed {} images in {:.2f} seconds. "
                            "{:.2f} images/second.".format(count, elapsed, count / elapsed))

# template_id = np.where(lmdb['up_index'] > 1)[0]  # NEVER use the last frame as template! I do not like bidirectional.
# rand_split = np.random.choice(len(template_id), len(template_id))
# lmdb['train_set'] = template_id[rand_split[:(len(template_id) - num_val)]]
# lmdb['val_set'] = template_id[rand_split[(len(template_id) - num_val):]]
# print(len(lmdb['train_set']))
# print(len(lmdb['val_set']))
#
# # to list for json
# lmdb['train_set'] = lmdb['train_set'].tolist()
# lmdb['val_set'] = lmdb['val_set'].tolist()
# lmdb['down_index'] = lmdb['down_index'].tolist()
# lmdb['up_index'] = lmdb['up_index'].tolist()
#
# print('lmdb json, please wait 5 seconds~')
# json.dump(lmdb, open('dataset.json', 'w'), indent=2)
print('done!')
