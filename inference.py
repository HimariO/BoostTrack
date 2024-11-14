import os
import cv2
import shutil
import time
from typing import *

import numpy as np
import dataset
import utils
from args import make_parser

import torch
from tqdm import tqdm
from default_settings import GeneralSettings, BoostTrackPlusPlusSettings, BoostTrackSettings
# from external.adaptors import detector
# from tracker.GBI import GBInterpolation
from boosttrack.tracker.boost_track import BoostTrack
from ultralytics import YOLO
from loguru import logger

"""
Script modified from Deep OC-SORT: 
https://github.com/GerardMaggiolino/Deep-OC-SORT
"""

DET_THRESH = 0.4

def load_video(video_path, sample_rate=None):
    assert os.path.exists(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_id = 0
    if sample_rate is None:
        sample_rate = max(1, fps // 4)
    
    frames = []
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            if frame_id % sample_rate == 0:
                frames.append(frame)
        else:
            # Break the loop if the end of the video is reached
            break
        frame_id += 1
    cap.release()
    
    transform = dataset.ValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    tensor_and_frame = []
    for f in tqdm(frames):
        ten, _ = transform(f, None, (f.shape[0], f.shape[1]))
        ten = torch.tensor(ten)[None, ...]
        # ten = torch.permute(ten, [0, 1, 2])
        # breakpoint()
        tensor_and_frame.append((ten, f))
    return tensor_and_frame


def vis_tracking(frames: List[np.ndarray], tracks: List[List[float]], video_path: str):
    
    # def write_frames(frames: List[np.ndarray], path):
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4/ BUG: this codec can't be open by web browser
    #     h, w, c = frames[0].shape
    #     writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    #     for frame in frames:
    #         writer.write(frame)
    #     writer.release()
    
    h, w, c = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4/ BUG: this codec can't be open by web browser
    writer = cv2.VideoWriter(video_path, fourcc, 10, (w, h))

    # Load an image
    for image, track in zip(frames, tracks):
        # Sample bounding box and label data
        # boxes = [(50, 50, 200, 200), (300, 100, 450, 300)]  # [(x1, y1, x2, y2), ...]
        # labels = ['Label 1', 'Label 2']
        labels = [f'_{int(t[4])}' for t in track]
        boxes = [t[:4] for t in track]

        # Iterate over the boxes and labels
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = [int(v) for v in box]

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Set label position above the top-left corner of the bounding box
            label_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)

            # Draw the label text
            cv2.putText(
                image,
                label,
                label_position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
        writer.write(image)
    writer.release()


def get_main_args():
    parser = make_parser()
    parser.add_argument("--video", type=str, default="mot17")
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--no_reid", action="store_true", help="mark if visual embedding should NOT be used")
    parser.add_argument("--no_cmc", action="store_true", help="mark if camera motion compensation should NOT be used")

    parser.add_argument("--s_sim_corr", action="store_true", help="mark if you want to use corrected version of shape similarity calculation function")

    parser.add_argument("--btpp_arg_iou_boost", action="store_true", help="BoostTrack++ arg. Mark if only IoU should be used for detection confidence boost.")
    parser.add_argument("--btpp_arg_no_sb", action="store_true", help="BoostTrack++ arg. Mark if soft detection confidence boost should NOT be used.")
    parser.add_argument("--btpp_arg_no_vt", action="store_true", help="BoostTrack++ arg. Mark if varying threhold should NOT be used for the detection confidence boost.")
    args = parser.parse_args()
    return args


@logger.catch
def main():
    # Set dataset and detector
    args = get_main_args()
    # GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = not args.no_cmc
    GeneralSettings.values['det_thresh'] = DET_THRESH
    GeneralSettings.values['aspect_ratio_thresh'] = 2
    GeneralSettings.values['max_age'] = 60
    # GeneralSettings.values['test_dataset'] = args.test_dataset

    BoostTrackSettings.values['s_sim_corr'] = args.s_sim_corr

    BoostTrackPlusPlusSettings.values['use_rich_s'] = not args.btpp_arg_iou_boost
    BoostTrackPlusPlusSettings.values['use_sb'] = not args.btpp_arg_no_sb
    BoostTrackPlusPlusSettings.values['use_vt'] = not args.btpp_arg_no_vt

    # detector_path, size = get_detector_path_and_im_size(args)
    detector_path = "external/weights/bytetrack_x_mot17.pth.tar"
    size = (800, 1440)
    # det = detector.Detector("yolox", detector_path, "mot17")
    yolo = YOLO('yolo11l')
    # print(det)
    

    tracker = None
    results = []
    target_map = []
    video_frames = []

    frame_count = 0
    total_time = 0
    video_name = 'output'
    
    # See __getitem__ of dataset.MOTDataset
    for frame_id, (img, np_img) in tqdm(enumerate(load_video(args.video, sample_rate=None))):
        img = img.cuda()

        # Initialize tracker on first frame of a new video
        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 0:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache()

            tracker = BoostTrack(video_name=video_name)

        tag = f"{video_name}:{frame_id}"
        # pred = det(img, tag)
        res = yolo.predict(np_img, conf=DET_THRESH)[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, np_img.shape[1])
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, np_img.shape[0])
        xyxy = torch.tensor(xyxy).cuda()

        bw = xyxy[:, 2] - xyxy[:, 0]
        bh = xyxy[:, 3] - xyxy[:, 1]
        mask = (bw > 10) & (bh > 10)
        # print('-' * 100)
        # print(mask)
        # print(xyxy)
        
        xyxy = xyxy[mask]
        conf = conf[mask]
        start_time = time.time()
        pred = torch.cat([xyxy, conf[:, None]], dim=-1)
        
        if not len(xyxy):
            continue
            
        # Nx6 of (x1, y1, x2, y2, ID, score)
        targets = tracker.update(pred, img, np_img, tag)
        tlwhs, ids, confs = utils.filter_targets(
            targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])

        total_time += time.time() - start_time
        frame_count += 1
        
        scale = min(img.shape[2] / np_img.shape[0], img.shape[3] / np_img.shape[1])
        targets[:4] *= scale
        
        results.append((frame_id, tlwhs, ids, confs))
        target_map.append(targets)
        video_frames.append(np_img)

    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    print(total_time)
    
    vis_tracking(video_frames, target_map, 'out.mp4')
    # Save detector results
    # det.dump_cache()
    # tracker.dump_cache()
    # # Save for all sequences
    # folder = os.path.join(args.result_folder, args.exp_name, "data")
    # os.makedirs(folder, exist_ok=True)
    # for name, res in results.items():
    #     result_filename = os.path.join(folder, f"{name}.txt")
    #     utils.write_results_no_score(result_filename, res)
    # print(f"Finished, results saved to {folder}")


if __name__ == "__main__":
    main()
