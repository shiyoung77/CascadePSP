import os
import json

import numpy as np
import matplotlib.cm as cm


output_path = os.path.join("/home/lsy/dataset/CoRL_real/metadata.json")

metadata = dict()
metadata['thing_classes'] = [
    "01_bleach_cleanser", "02_mug", "03_bowl", "04_potted_meat_can", "05_tomato_can", "06_large_marker", "07_banana",
    "08_pudding_box", "09_extra_large_clamp", "10_mustard_bottle", "11_pitcher_base", "12_master_chef_can",
    "13_dumbbell", "14_duck", "15_clorox", "16_lamp", "17_detergent", "18_dinosaur", "19_Charmander", "20_power_drill"
]

cmap = cm.get_cmap('tab20')
colors = [(np.array(color) * 255).astype(int).tolist() for color in cmap.colors]
metadata['thing_colors'] = colors

with open(output_path, 'w') as fp:
    json.dump(metadata, fp, indent=4)
