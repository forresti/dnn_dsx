
#CAFFE_ROOT=/home/eecs/forresti/caffe_depthMax_and_hog
CAFFE_ROOT=$MEMBERWORK/csc103/dnn_exploration/caffe-bvlc-master

#TODO: divide by # iterations. (or, switch to Caffe dev branch)
for d in ./nets/*
do
  echo $d
  #$CAFFE_ROOT/build/tools/caffe time -model=$n -gpu=0 -iterations 10
  #$CAFFE_ROOT/build/tools/caffe time -model=$d/deploy.prototxt -iterations=10
  #aprun -n 1 -d 16 $CAFFE_ROOT/build/tools/caffe time -model=$d/deploy.prototxt -iterations=10 -gpu=0

  #$CAFFE_ROOT/build/tools/caffe time -model=$d/deploy.prototxt -iterations=10 > $d/timing.log 2>&1 
  aprun -n 1 -d 16 $CAFFE_ROOT/build/tools/caffe time -model=$d/deploy.prototxt -iterations=10 -gpu=0 > $d/timing.log 2>&1 

done
