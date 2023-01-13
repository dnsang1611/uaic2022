# This code splits rec_results.txt file (saved in parseq) into multiple imXXXX.txt files and save them in parseq-submit folder
import os
import glob 

PARSEQ_SUBMIT_PATH = 'parseq-submit'

# Remove old files to write append
for old_file in glob.glob(PARSEQ_SUBMIT_PATH + '/*'):
    os.remove(old_file)


with open('../parseq/rec_result.txt') as f:
    rows = f.readlines()

for row in rows:
    # im_name_bbox: im0000_0001.jpg
    # label: ...

    im_name_bbox  = row.split('\t')[0]
    label = row.split('\t')[1]
    # im_name: im0000
    im_name = im_name_bbox.split('.')[0].split('_')[0]

    f_write = open(f'{PARSEQ_SUBMIT_PATH }/{im_name}.txt', 'a')
    f_write.write(label)
f.close()