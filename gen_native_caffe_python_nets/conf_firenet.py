

#platform='aspire'
platform='titan'

#dataset='places2'
dataset='imagenet'

#paths for train/test data
train_lmdb='dummy_path'
test_lmdb='dummy_path'

if platform=='aspire':
    if dataset=='imagenet':
        train_lmdb='/nscratch/forresti/ILSVRC12/ilsvrc2012_train_256x256_lmdb/'
        test_lmdb='/nscratch/forresti/ILSVRC12/ilsvrc2012_val_256x256_lmdb/'
    elif dataset=='places2':
        train_lmdb='/nscratch/forresti/datasets/places2_train_lmdb/'
        test_lmdb='/nscratch/forresti/datasets/places2_val_lmdb/'
    else:
        print 'unknown dataset'

if platform=='titan':
    if dataset=='imagenet':
        train_lmdb="/lustre/atlas/scratch/forresti/csc103/dnn_exploration/datasets/ilsvrc2012_train_256x256_lmdb"
        test_lmdb="/lustre/atlas/scratch/forresti/csc103/dnn_exploration/datasets/ilsvrc2012_val_256x256_lmdb"
    elif dataset=='places2':
        train_lmdb="/lustre/atlas/scratch/forresti/csc103/dnn_exploration/datasets/places2_train_lmdb"
        test_lmdb="/lustre/atlas/scratch/forresti/csc103/dnn_exploration/datasets/places2_val_lmdb"
    else:
        print 'unknown dataset'


