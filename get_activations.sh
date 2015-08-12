
caffe_root=/home/eecs/forresti/__caffe_allreduce
now=`date +%a_%Y_%m_%d__%H_%M_%S`
g=$2
d=$1 #you pass in directory of net 
#d=nets_multiGPU/NiN_batch256_stepsize50k_maxiter110k_fromCompiler_withDropout_fixWeightInit
weights=`ls -t $d/*caffemodel | head -1`
#iter=782 # ceil(50000/64)
iter=1563 #50000/32
#output_blob=pool4 #TODO: auto-select final blob
#output_blob=fc8
output_blob=prob
#output_db_path=$d/activations_lmdb
output_db_path=$d/activations_prob_lmdb

rm -rf $output_db_path

#Usage: extract_features  pretrained_net_param  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type  [CPU/GPU] [DEVICE_ID=0]

#$caffe_root/build/tools/extract_features $weights $d/trainval.prototxt fc8 $output 35 lmdb GPU 0

CUDA_VISIBLE_DEVICES=$g $caffe_root/build/tools/extract_features $weights $d/trainval.prototxt $output_blob \
  $output_db_path $iter lmdb GPU 0 \
  > $d/get_activations_$now.log 2>&1 &

