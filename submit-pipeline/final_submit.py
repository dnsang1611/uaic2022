# This code merges data from parseq-submit and mmocr-submit then makes predicted folder

import os

BDB_DIR = 'mmocr-submit'
LABEL_DIR = 'parseq-submit'
SUBMIT_DIR = 'predicted'

if not os.path.exists(SUBMIT_DIR):
    os.mkdir(SUBMIT_DIR)
    
for iname in sorted(os.listdir(BDB_DIR)):
    # bdb
    with open(BDB_DIR + '/' + iname) as f:
        bdboxes = f.readlines()

    if len(bdboxes) == 0:
        with open(SUBMIT_DIR + '/' + iname, 'w') as f:
            f.write('0,0,0,0,0,0,0,0,###')
    else:
        with open(LABEL_DIR + '/' + iname) as f:
            labels = f.readlines()
        
        with open(SUBMIT_DIR + '/' + iname, 'w') as f:
            n_labels = len(labels)
            for i in range(n_labels):
                bdbox = bdboxes[i].split('\n')[0]
                label = labels[i].split('\n')[0]
                f.write(f'{bdbox}{label}')

                if i != (n_labels -1):
                    f.write('\n')

