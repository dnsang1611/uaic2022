dataset_type = 'IcdarDataset'
data_root = '/workspace/uaic22-data/uaic2022_training_data_dbnetpp' #'data/icdar2015'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_train.json',
    img_prefix=f'{data_root}/images',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/images',
    pipeline=None)

train_list = [train]

test_list = [test]
