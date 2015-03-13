from parse_logs_custom import get_latest_log
from parse_logs_custom import get_current_accuracy
import os 
from IPython import embed
from shutil import rmtree

'''
Forrest's logo experiment notes:
- step 1: done 3/12/15
- step 2: done 3/12/15
- step 3: TODO
- step 4: TODO
'''

if __name__ == "__main__":

    #make sure you're ok messing up baseDir. (keep a backup.)
    baseDir = '/nscratch/forresti/dsx_backup/nets_backup_1-9-15_LOGO'

    #step 1: identify "over 30% imagenet accuracy" nets; delete the others.
    all_nets = os.listdir(baseDir) #0, 1, 2, ...    
    #print all_nets
    #all_nets = ["341"]
    for netDir in all_nets:
        try:
            log_filename = get_latest_log(baseDir + '/' + netDir)
            accuracy_dict = get_current_accuracy(log_filename) 
        except:
            #continue
            accuracy_dict = "error"

        if accuracy_dict is "error":
            print "removing net: ", netDir,' due to lack of accuracy data'
            rmtree(baseDir + '/' + netDir, ignore_errors=True) #ignore errors to delete non-empty directory

        elif accuracy_dict['accuracy'] >= 0.3:
            print "preserving net: ",netDir, ' accuracy = ',accuracy_dict['accuracy']

        else: #accuracy < 0.3
            print "removing net: ", netDir, ' accuracy = ',accuracy_dict['accuracy']
            rmtree(baseDir + '/' + netDir, ignore_errors=True)

    #TODO: make this *actually* remove useless directories.

    #step 2: find/replace dataset path to "FlickrLogos-32" in the prototxt files 
    # (you do this yourself in bash...)
    '''
        bash commands to find/replace dataset path:
    sed -i 's/\/lustre\/atlas\/scratch\/forresti\/csc103\/dnn_exploration\/caffe-bvlc-master\/data\/ilsvrc12\/imagenet_mean.binaryproto/\/home\/eecs\/forresti\/caffe_depthMax_and_hog\/data\/ilsvrc12\/imagenet_mean.binaryproto/g' 1/trainval.prototxt

    sed -i 's/\/lustre\/atlas\/scratch\/forresti\/csc103\/dnn_exploration\/ilsvrc2012_train_256x256_lmdb/\/nscratch\/forresti\/FlickrLogos-32\/FlickrLogos-32_trainval_logosonly_lmdb/g' 1/trainval.prototxt

    sed -i 's/\/lustre\/atlas\/scratch\/forresti\/csc103\/dnn_exploration\/ilsvrc2012_val_256x256_lmdb/\/nscratch\/forresti\/FlickrLogos-32\/FlickrLogos-32_test_logosonly_lmdb/g' 1/trainval.prototxt

    #might do:
    sed -i "s/batch_size: 256/batch_size: 32/g" 1/trainval.prototxt #train
    sed -i "s/batch_size: 50/batch_size: 25/g" 1/trainval.prototxt #val

    ''' 

    #step 3: add solver.prototxt
    # (do this yourself?)

    '''
    created generic solver here: /nscratch/forresti/dsx_backup/nets_backup_1-9-15_LOGO/solver.prototxt

    automated:
    for d in /nscratch/forresti/dsx_backup/nets_backup_1-9-15_LOGO/*
    do
        echo $d
        #rm $d/solver.prototxt
        ln -s /nscratch/forresti/dsx_backup/nets_backup_1-9-15_LOGO/solver.prototxt $d/solver.prototxt
    done
    '''

    #step 4: finetune the nets on logo recognition (on a19 and a20)
    # (do this yourself using train_nets.sh or similar)

    '''
    cd /nscratch/forresti/dsx_backup/nets_backup_1-9-15_LOGO/0
    $CAFFE_ROOT/build/tools/caffe train -solver=solver.prototxt -weights=caffe_train_iter_305000.caffemodel -gpu=0 > train_$now.log 2>&1 &  
    '''

