import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument('--mot', type=str, default=None, help='name of mot for saving detections')
    parser.add_argument('--det_path', type=str, default=None, help='name of path for loading saved detections detections')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def load_dets(path):
    det_dict = {}
    with open(path, "r") as filestream:
        for line in filestream:
            detection = line.split(",")
            frame_id = int(detection[0])
            x1 = float(detection[2])
            y1 = float(detection[3])
            w  = float(detection[4])
            h  = float(detection[5])
            s  = float(detection[6])
            x2 = x1 + w
            y2 = y1 + h

            # Load into list
            det = [x1, y1, x2, y2, s]

            if frame_id in det_dict:
                id_list = det_dict[frame_id]
                id_list.append(det)
                det_dict[frame_id] = id_list
            else:
                det_dict[frame_id] = [det]

    return det_dict

def detection_demo(save_folder, args):
    # Load detections from text file to be used as input for bytetrack
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps, scale=False)
    timer = Timer()
    results = []

    # load the detection results into a dictionary
    det_dict = load_dets(args.det_path)
    logger.info("Detections successfully loaded from {}".format(args.det_path))

    for frame_id, img_path in enumerate(files, 1):
        if frame_id in det_dict:
            # Run bytetracker
            online_targets = tracker.update(np.array(det_dict[frame_id]), [0, 0], [0, 0])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
        else:
            timer.toc()

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(save_folder, f"{args.mot}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(args):
    if not args.experiment_name:
        args.experiment_name = args.mot

    output_dir = "./bytetrack_output"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Args: {}".format(args))

    # Run detection demo
    detection_demo(output_dir, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
