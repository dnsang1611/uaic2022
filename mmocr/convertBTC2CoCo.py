"""
file: {
    "images": [
        {
            "file_name": ...,
            "height": ...,
            "width": ...,
            "id": ..., 
        },
        {
            "file_name": ...
            ...
        }
    ],

    "categories: [
        {
            "id": 1
            "name": "text"
        }
    ],

    annotations: [
        {
            "iscrowd": 1, 
            "category_id": 1, 
            "bbox": [260.0, 138.0, 24.0, 20.0], 
            "area": 402.0, 
            "segmentation": [[261, 138, 284, 140, 279, 158, 260, 158]], 
            "image_id": 0, 
            "id": 0
        },
        {
            "iscrowd": 0, 
            "category_id": 1, 
            "bbox": [288.0, 138.0, 129.0, 23.0], 
            "area": 2548.5, 
            "segmentation": [[288, 138, 417, 140, 416, 161, 290, 157]], 
            "image_id": 0, 
            "id": 1}
    ]
}
"""

import numpy as np
import os

LABELS_PATH = '/workspace/uaic22-data/uaic2022_training_data_dbnetpp/labels'
IMAGES_PATH = '/workspace/uaic22-data/uaic2022_training_data_dbnetpp/images'
TEST_PATH = '/workspace/uaic22-data/uaic2022_training_data_dbnetpp/instances_test.json'
TRAIN_PATH = '/workspace/uaic22-data/uaic2022_training_data_dbnetpp/instances_train.json'

# Shoelace method
def find_area(polygon):
    polygon = np.array(polygon)
    x = polygon[:, 0]
    y = polygon[:, 1]

    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)

    return float(area)

def find_bbox(polygon):
    polygon = np.array(polygon)
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    x_min = float(x_min)
    x_max = float(x_max)
    y_max = float(y_max)
    y_min = float(y_min)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

infor_train = {
    "images": [],
    'categories': [{"id": 1, "name": "text"}],
    "annotations" : []
}

infor_test = {
    "images": [],
    'categories': [{"id": 1, "name": "text"}],
    "annotations" : []
}

import json 
import cv2

paths = os.listdir(LABELS_PATH)
num_imgs = len(paths)
val_ratio = 0.2

val_idx_end = int(num_imgs*(val_ratio))
count = 0
temp = 0
for id, path in enumerate(paths):
    if (id + 1) % 200 == 0:
        print(id + 1)

    with open(f'{LABELS_PATH}/{path}') as f:
        words = json.load(f)

    file_name = path.split('.')[0] + '.jpg'
    file_id = int(path.split('.')[0].split('im')[1])
    height, width, channels = cv2.imread(f'{IMAGES_PATH}/{file_name}').shape

    image = {
            'file_name': file_name,
            'height': height,
            'width': width,
            'id': file_id, 
        }

    if id < val_idx_end:
        # file_name = f'test/{file_name}'
        # image = {
        #     'file_name': file_name,
        #     'height': height,
        #     'width': width,
        #     'id': file_id, 
        # }
        infor_test['images'].append(image)
    else:
        # file_name = f'train/{file_name}'
        # image = {
        #     'file_name': file_name,
        #     'height': height,
        #     'width': width,
        #     'id': file_id, 
        # }
        infor_train['images'].append(image)

    for word in words:
        polygon = word['points']
        bbox = find_bbox(polygon)
        area = find_area(polygon)
        seg = []
        for i in range(len(polygon)):
            seg.append(polygon[i][0])
            seg.append(polygon[i][1])

        object = {
            'iscrowd': 0,
            'category_id': 1,
            'bbox': bbox,
            'area': area,
            'segmentation': [seg],
            'image_id': file_id,
            'id': count
        }

        if id < val_idx_end:
            infor_test['annotations'].append(object)
        else:
            infor_train['annotations'].append(object)
        
        count += 1
    # if temp == 2:
    #     break
    # temp += 1
with open(TRAIN_PATH, 'w') as wf:
    json.dump(infor_train, wf, indent=5)

with open(TEST_PATH, 'w') as wf:
    json.dump(infor_test, wf, indent=5)