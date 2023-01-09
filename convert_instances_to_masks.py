import os
import json
import argparse

import numpy as np
import pycocotools.mask
from PIL import Image
from tqdm import tqdm

from add_palette_to_mask import _palette


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='/home/lsy/dataset/CoRL_real')
    parser.add_argument("-v", "--videos", type=str, default=["0001"], nargs="+")
    parser.add_argument("--annotation_folder", type=str, default="annotations/instances")
    parser.add_argument("--output_folder", type=str, default="annotations/refined_masks")
    args = parser.parse_args()

    args.videos = [f"{i:04d}" for i in range(1, 31)]
    for video in args.videos:
        video_folder = os.path.join(args.dataset, video)
        annotation_folder = os.path.join(video_folder, args.annotation_folder)
        annotation_files = sorted([f for f in os.listdir(annotation_folder) if f.endswith('json')])

        output_folder = os.path.join(video_folder, args.output_folder)
        os.makedirs(output_folder, exist_ok=True)

        for annotation_file in tqdm(annotation_files):
            annotation_path = os.path.join(annotation_folder, annotation_file)
            with open(annotation_path, 'r') as fp:
                metadata = json.load(fp)

            annotations = metadata['annotations']
            im_h, im_w = metadata['height'], metadata['width']
            combined_mask = np.zeros((im_h, im_w), dtype=np.uint8)  # category id must not exceed 255

            for annotation in annotations:
                category_id = annotation['category_id']  # [0, N)
                assert category_id < 256, "category id must not exceed 255"
                rle = annotation['segmentation']
                mask = pycocotools.mask.decode(rle).astype(bool)
                combined_mask[mask] = category_id + 1  # 0 means background

            prefix = annotation_file.split('-')[0]
            combined_mask = Image.fromarray(combined_mask).convert('P')
            combined_mask.putpalette(_palette)
            combined_mask.save(os.path.join(output_folder, f"{prefix}-mask.png"))


if __name__ == "__main__":
    main()
