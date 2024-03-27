import pandas as pd
import numpy as np


# Mean RGB color values
MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673], dtype=np.float32)

# Labels for RfS dataset
RFS_labels = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 'kitchen_cabinet', 'display', 'trash_bin', 'other_shelf', 'other_table']

# Labels for CAD dataset
CAD_labels = ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']

# Counts of each label in CAD dataset
CAD_cnts = [555, 1093, 212, 113, 232, 260, 191, 121]

# Weights for each label in CAD dataset, calculated as the total count divided by the count for each label
CAD_weights = np.sum(CAD_cnts) / np.array(CAD_cnts)

# Mapping from CAD labels to ShapeNet IDs
CAD2ShapeNetID = ['4379243', '3001627', '2871439', '4256520', '2747177', '2933112', '3211117', '2808440']

# Mapping from indices to selected categories from SHAPENETCLASSES
CAD2ShapeNet = {k: v for k, v in enumerate([1, 7, 8, 13, 20, 31, 34, 43])}

# Inverse mapping from CAD2ShapeNet
ShapeNet2CAD = {v: k for k, v in CAD2ShapeNet.items()}

# List of labels that should not fly. cabinet, display, and bathtub (sink) may fly.
CADNotFly = [0, 1, 2, 3, 4]

# File path to the label map file
raw_label_map_file = 'datasets/scannet/rfs_label_map.csv'

# Read the label map file into a pandas DataFrame
raw_label_map = pd.read_csv(raw_label_map_file)

# Initialize a dictionary to map from RFS labels to CAD labels
RFS2CAD = {}

# Populate the RFS2CAD dictionary
for i in range(len(raw_label_map)):
    row = raw_label_map.iloc[i]
    RFS2CAD[int(row['rfs_ids'])] = row['cad_ids']

# Initialize an array to map from RFS labels to CAD labels
RFS2CAD_arr = np.ones(30) * -1

# Populate the RFS2CAD_arr array
for k, v in RFS2CAD.items():
    RFS2CAD_arr[k] = v
