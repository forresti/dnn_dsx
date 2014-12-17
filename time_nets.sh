
CAFFE_ROOT=/home/eecs/forresti/caffe_depthMax_and_hog
for n in ./nets/*
do
  echo $n
  $CAFFE_ROOT/build/tools/caffe time -model=$n -gpu=0 -iterations 10
done
