
#do these one at a time for now.

for d in /nscratch/forresti/dsx_backup/nets_backup_1-9-15_LOGO/*
do
    echo $d
    cd $d
    now=`date +%a_%Y_%m_%d__%H_%M_%S`
    newestWeights=`ls -t $d/*caffemodel | head -1` #thx: stackoverflow.com/questions/5885934
    $CAFFE_ROOT/build/tools/caffe train -solver=solver.prototxt -weights=$newestWeights -gpu=0 > finetune_$now.log 2>&1 
done

