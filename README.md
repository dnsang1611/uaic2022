# UIT AI CHALLANGE 2022 - TEAM: HERMES
## 1. Create container
#### DBNetpp:
`docker run --mount type=bind,source={path/to/uaic2022_submit},target=/workspace/ --name Hermes_uaic2022_sangdn_dpnetpp -it --gpus all --cpus 20 --shm-size=2gb  21522542/mmocr_hermes:v1`
#### PARSEQ:
`docker run --mount type=bind,source={path/to/uaic2022_submit},target=/workspace/ --name Hermes_uaic2022_sangdn_parseq -it --gpus all --cpus 20 --shm-size=2gb  21522542/parseq_hermes:v1`

If you don’t want to train models, you can go straight to 6th section to infer

## 2. Prepare dataset
### Overview:
- uaic2022_training_data_dbnetpp: includes preprocessed data
- uaic2022_training_data_parseq: includes preprocessed data and augmented data (images with id >= 5000 are augmented images)
### Convert labels:
#### DBNetpp: 
we need to convert original labels to COCO format. With mmocr/convertBTC2CoCo.py file, we created instances_test.json and instances_train.json which follow the COCO format.
#### PARSEQ: 
we need to convert labels and images to mdb.file. To do this, run following commands in order
python prepare_data_for_mdb.py
python tools/create_lmdb_dataset.py data/cropped-images/ train_gt_file.txt train/sin_hw
python tools/create_lmdb_dataset.py data/cropped-images/ val_gt_file.txt val/sin_hw
		After that, we will see structure like this:

## 3. Train DBNetpp:
- Download pretrained model on English: https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth
- Train: In dbnetpp container, cd /workspace/mmocr
- Single GPU:
`CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py --work-dir {work-dir} --load-from {path/to/pretrained/model}`
- Multiple GPU:
`tools/dist_train.sh configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py dbnetpp {n_gpus} --load-from {path/to/pretrained/model/of/mmocr}
Our checkpoints, log will be  saved folder {work-dir}`

## 4. Train PARSEQ: In parseq container, cd /workspace/parseq
- Specify number of gpus: uaic2022_submit/parseq/configs/main.yaml → trainer → gpus
- Example: use 3 gpus to train
- Train: python3 train.py pretrained=parseq 
- Note: pretrained=parseq will load pretrained model from this link: https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt
- Our checkpoints, log will be saved in folder outputs

## 5. Inference
- In source code, we downloaded the private test. You can remove this folder and download again if you want
- Our pretrained models: We have 2 pretrained models placed in pretrained-models folder. These models are also available on google drive
- DBNetpp: 	https://drive.google.com/file/d/1nI9JH-6ND25TXxd_tcyQSUxsXJpb0aG3/view?usp=share_link

- PARSEQ: https://drive.google.com/file/d/1cWtdXBkHn6p-zMnF37B7DYdSDqAg47yl/view?usp=share_link

### In DBNetpp container, run following commands:
- Commands:
`cd /workspace/mmocr`
`python mmocr/utils/ocr.py {path/to/images/folder} --det DBPP_r50 --det-ckpt {path/to/pretrained/model} --recog None --export inference_results/ --output inference_results/	`
- Example:
`python mmocr/utils/ocr.py /workspace/uaic22-data/uaic2022_private_test/images --det DBPP_r50 --det-ckpt /workspace/pretrained-model/dbnetpp/best_0_hmean-iou:hmean_epoch_700.pth --recog None --export inference_results/ --output inference_results/`
- The results will be saved in inference_results → make sure this folder is empty before running this command

### In PARSEQ container, run following commands:
- Commands:
`cd /workspace/submit-pipeline`
`python extract_data_from_mmocr.py`
- Change RAW_IMG_PATH (line 27) in this file if necessary
- Change threshold (in line 66) if necessary. This will help us drop bounding boxes which have score < threshold. In private test, we use threshold = 0.6
- Results will saved in 3 folders: dropped-images (includes cropped images which have score < threshold), mmocr-submit (includes bounding box files), parseq-cropped-images (includes cropped images will be passed to PARSEQ to infer)
- Make sure 3 above folders are empty before running

`cd /workspace/parseq
python3 /workspace/parseq/read.py {path/to/pretrained/model} --images /workspace/submit-pipeline/parseq-cropped-images/*`
- Example:
`python3 /workspace/parseq/read.py /workspace/pretrained-model/parseq/epoch=79-step=9230-val_accuracy=88.3849-val_NED=95.1411.ckpt --images /workspace/submit-pipeline/parseq-cropped-images/*`
- Results will be saved in /parseq/rec_result.txt

`cd /workspace/submit-pipeline`
`python extract_data_from_parseq.py`
- Results will be saved in folder parseq-submit
`python final_submit.py`
- Results will be saved in folder predicted
`zip -r predicted.py predicted`
