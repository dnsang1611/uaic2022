# def shoelace_formula(polygonBoundary, absoluteValue = True):
#     nbCoordinates = len(polygonBoundary)
#     nbSegment = nbCoordinates - 1

#     l = [(polygonBoundary[i+1][0] - polygonBoundary[i][0]) * (polygonBoundary[i+1][1] + polygonBoundary[i][1]) for i in range(nbSegment)]

#     if absoluteValue:
#         return abs(sum(l) / 2.)
#     else:
#         return sum(l) / 2.

# polygon = [[5, 0], [6, 4], [4, 5], [1, 5], [1, 0]]
# print(shoelace_formula(polygon))
# This file makes bbox file (saved in mmocr-submit) and crop image (saved in crop images) for parseq to infer 

# Note: Đức viết code này, nhớ đặt tên sub image theo định dạng im0000_0001.jpg, đừng đặt
# theo dạng im0000_1.jpg, không là như test 1 :v

import numpy as np
import cv2
import os
import glob
import json 

INFER_PATH = '../mmocr/inference_results'
CROPPED_IMG_PATH = 'parseq-cropped-images'
RAW_IMG_PATH = '../uaic22-data/uaic2022_private_test/images' # need to change
BBOX_PATH = 'mmocr-submit'
DROP_PATH = 'dropped-images'

# Crop bbox
def crop_image(img , polygon):
    ## (1) Crop the bounding rect
    polygon = np.array(polygon)
    rect = cv2.boundingRect(polygon)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    ## (2) make mask
    polygon = polygon - polygon.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)
    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    return dst

# Calculate the area of the polygon
def find_area(polygon):
    polygon = np.array(polygon)
    x = polygon[:, 0]
    y = polygon[:, 1]

    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)

    return area

def relu(x):
    return x if x >= 0 else 0

count_drop = 0
count_used = 0
count_undetected = 0
threshold = 0.6

for path in glob.glob(f'{INFER_PATH}/*.json'):
    with open(path) as f:
        content = json.load(f)
    
    poly_scr_pairs = content['boundary_result']
    img_name = os.path.basename(path).split('_')[1].split('.')[0]
    count = 0
    img = cv2.imread(f'{RAW_IMG_PATH}/{img_name}.jpg')
    # open file txt
    wf = open(f'{BBOX_PATH}/{img_name}.txt', 'w') 

    if len(poly_scr_pairs) == 0: # No polygon is detected in image
        count_undetected += 1

        height, width, channels = img.shape
        wf.write(f'0,0,{width},0,{width},{height},0,{height}')

        croped_path = f'{CROPPED_IMG_PATH}/{img_name}_{count:04n}.jpg'
        cv2.imwrite(croped_path, img)
    else:
        # poly_scr_pairs: list of bounding boxes
        # poly_scr_pair : a single bounding box
        for poly_scr_pair in poly_scr_pairs:
            # Write to file in mmocr-submit
            length = len(poly_scr_pair) - 1 # Drop score
            scr = poly_scr_pair[-1]
            polygon = [[relu(round(poly_scr_pair[i])), relu(round(poly_scr_pair[i + 1]))] for i in range(0, len(poly_scr_pair) - 1, 2) ]        

            if scr >= threshold:
                for i in range(length):
                    wf.write(str(relu(round(poly_scr_pair[i]))))
                    if i != length:
                        wf.write(',')
                wf.write('\n')
                
            # Crop image
            cropped_img = crop_image(img, polygon)
            if scr >= threshold:
                croped_path = f'{CROPPED_IMG_PATH}/{img_name}_{count:04n}.jpg'
                count_used += 1
            else:
                croped_path = f'{DROP_PATH}/{img_name}_{count:04n}.jpg'
                count_drop += 1

            cv2.imwrite(croped_path, cropped_img)

            count += 1

    wf.close()

print(f'Total: {count_drop + count_used + count_undetected}')
print(f'No used images: {count_used}') # images which will be used to pass into parseq
print(f'No dropped images: {count_drop}') # images which will be dropped
print(f'No undetected images: {count_undetected}') # images which are not detected by dbnetpp