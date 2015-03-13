
#do these one at a time for now.

$CAFFE_ROOT/build/tools/caffe train -solver=solver.prototxt -weights=caffe_train_iter_305000.caffemodel -gpu=0 > train_$now.log 2>&1 

