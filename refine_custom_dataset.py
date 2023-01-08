import os
import json
import argparse

import numpy as np
import cv2
import torch
import pycocotools
import matplotlib.pyplot as plt
import segmentation_refinement as refine  # pip install segmentation-refinement
from tqdm import tqdm
from PIL import Image
from detectron2.utils.visualizer import Visualizer, ColorMode, BoxMode
from detectron2.data import MetadataCatalog


class CustomVisualizer(Visualizer):

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]

            colors = None
            category_ids = [x["category_id"] for x in annos]
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    [x / 255 for x in self.metadata.thing_colors[c]]  # DO NOT JITTER !!!
                    for c in category_ids
                ]
            names = self.metadata.get("thing_classes", None)
            labels = [names[i] for i in category_ids]
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )
        return self.output

    def _change_color_brightness(self, color, brightness_factor):
        modified_color = super()._change_color_brightness(color, brightness_factor)
        modified_color = tuple(color if color < 1 else 1 for color in modified_color)
        return modified_color


def mask_to_rle(mask) -> dict:
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    assert isinstance(mask, np.ndarray)
    rle = pycocotools.mask.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def refine_video(video_path, stride=1):
    refiner = refine.Refiner(device='cuda:0')  # device can also be 'cpu'

    output_folder = os.path.join(video_path, 'annotations', 'instances')
    os.makedirs(output_folder, exist_ok=True)

    color_files = sorted(os.listdir(os.path.join(video_path, 'color')))
    for i in tqdm(range(0, len(color_files), stride)):
        color_file = color_files[i]
        prefix = color_file.split('-')[0]
        color_path = os.path.join(video_path, 'color', color_file)
        bgr_im = cv2.imread(color_path)

        mask_path = os.path.join(video_path, 'annotations', 'masks', f'{prefix}-mask.png')
        mask = np.asarray(Image.open(mask_path))
        # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        instance = dict()
        instance['file_name'] = os.path.join(os.path.basename(video_path), 'color', color_file)
        instance['height'] = bgr_im.shape[0]
        instance['width'] = bgr_im.shape[1]
        instance['annotations'] = []

        unique_ids = np.unique(mask).tolist()
        for id in reversed(unique_ids):
            if id == 0:  # background
                continue

            obj_mask = (mask == id).astype(np.uint8) * 255
            # refined_mask = refiner.refine(bgr_im, obj_mask, fast=True)
            # refined_mask[refined_mask < 220] = 0
            # refined_mask = refiner.refine(bgr_im, refined_mask, fast=True)
            # refined_mask[refined_mask < 220] = 0
            # refined_mask = refiner.refine(bgr_im, refined_mask, fast=False)
            refined_mask = refiner.refine(bgr_im, obj_mask, fast=False)
            refined_mask = refined_mask.astype(bool)

            ys, xs = refined_mask.nonzero()
            if len(ys) == 0:
                continue

            box = np.array([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])

            ann = dict()
            ann['bbox'] = box.tolist()
            ann['bbox_mode'] = 0  # BoxMode.XYXY_ABS
            ann['category_id'] = id - 1  # detectron2 requires id from 0 ~ num_cat - 1
            ann['segmentation'] = mask_to_rle(refined_mask)  # RLE format
            instance['annotations'].append(ann)

        with open(os.path.join(output_folder, f'{prefix}-ann.json'), 'w') as fp:
            json.dump(instance, fp)


def visualize_video_annotations(video_path, metadata):
    color_files = sorted(os.listdir(os.path.join(video_path, 'color')))
    annotation_folder = os.path.join(video_path, 'annotations', 'instances')
    for i in tqdm(range(0, len(color_files), 10)):
        color_file = color_files[i]
        prefix = color_file.split('-')[0]
        color_path = os.path.join(video_path, 'color', color_file)
        bgr_im = cv2.imread(color_path)
        rgb_im = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB)

        annotation_path = os.path.join(annotation_folder, f'{prefix}-ann.json')
        with open(annotation_path, 'r') as fp:
            instance = json.load(fp)

        visualizer = CustomVisualizer(rgb_im, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
        out = visualizer.draw_dataset_dict(instance)
        vis_rgb_im = out.get_image()
        vis_bgr_im = cv2.cvtColor(vis_rgb_im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(annotation_folder, f'{prefix}-vis.jpg'), vis_bgr_im)
        cv2.imshow("annotation", vis_bgr_im)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='/home/lsy/dataset/CoRL_real')
    parser.add_argument("-v", "--videos", type=str, default=["0001"], nargs="+")
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    with open(os.path.join(args.dataset, 'metadata.json'), 'r') as fp:
        meta = json.load(fp)
    metadata = MetadataCatalog.get("CoRL_real")
    metadata.set(thing_classes=meta['thing_classes'])
    metadata.set(thing_colors=meta['thing_colors'])

    # args.videos = [f"{i:04d}" for i in range(1, 31)]
    for video in args.videos:
        video_path = os.path.join(args.dataset, video)
        refine_video(video_path, args.stride)
        visualize_video_annotations(video_path, metadata)


if __name__ == "__main__":
    main()
