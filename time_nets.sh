
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

#timing results:
grep "Forward pass:" ./nets/*/timing.log
    #Forward pass ranges from 650ms to 4750ms 
    #and 2 outliers: 
    #    [seed 7: 8692ms] -- [layer0_conv: kernel=3, stride=4]->...->[layer2_conv: kernel=3, stride=1]->...->[layer6_conv: kernel=7, stride=1] 
    #    [seed 48: 6579ms] -- no pooling, small conv strides

#time AlexNet:
aprun -n 1 -d 16  $CAFFE_ROOT/build/tools/caffe time -model=$CAFFE_ROOT/models/bvlc_alexnet/deploy_batch256.prototxt -gpu=0 -iterations=10
    #Forward pass: 5774.85 milliseconds. 
    #as with the results above, this is sum of 10 iterations' time. (timing isn't averaged in Caffe-master)


#errors:
grep "error" ./nets/*/timing.log

