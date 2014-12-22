
CAFFE_ROOT=$MEMBERWORK/csc103/dnn_exploration/caffe-bvlc-master

#aprun -n 1 -d 16 $CAFFE_ROOT/build/tools/caffe train -solver=./nets/0/solver.prototxt -gpu=0 

#for d in ./nets/*
for((i=0; i<150; i++))
do
  d=./nets/$i
  #echo $d
  now=`date +%a_%Y_%m_%d__%H_%M_%S`
  #aprun -n 1 -d 16 $CAFFE_ROOT/build/tools/caffe train -solver=$d/solver.prototxt -gpu=0 > $d/train_$now.log 2>&1 &
  aprun -n 1 -d 16 $CAFFE_ROOT/build/tools/caffe train -solver=$d/solver.prototxt -snapshot=$d/caffe_train_iter_20000.solverstate -gpu=0 > $d/train_$now.log 2>&1 &
   #if [ -f $d/caffe_train_iter_10000.solverstate ]
   #then
   #    echo $d/caffe_train_iter_10000.solverstate
   #fi
done

#example training:
#$CAFFE_ROOT/build/tools/caffe train -solver=nets/0/solver.prototxt 
#aprun -n 1 -d 16 $CAFFE_ROOT/build/tools/caffe train -solver=nets/0/solver.prototxt 


#grep "Test net output #0: accuracy =" ./nets/*/train_Sun_2014_12_21_16* 

