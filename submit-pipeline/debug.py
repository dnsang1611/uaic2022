import json 
with open('../PaddleOCR-Vietnamese/uaic22-data/uaic2022_training_data/instances_train.json') as f:
    annotations = json.load(f)['annotations']

st = set()

for annotation in annotations:
    polygon = annotation['segmentation'][0]
    st.add(len(polygon))

print(st)
