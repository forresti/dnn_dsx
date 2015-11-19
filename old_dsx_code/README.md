
rough workflow:

1. generate DNNs
./gen_nets.sh 
    calls random_net_generator.py and solver_generator.py
    outputs to ./nets/$seed

2. time DNNs
./time_net.sh
    [TODO] copies nets that are sufficiently fast to [some folder]

3. train DNNs (many concurrent training runs on compute cluster)
#quick-and-dirty:
./train_nets.py #spawns many aprun jobs

#or, use Python (YMMV... we have had trouble with subprocess.call() + MPI)
#python ./mpi_dnn.py

#most usable at scale -- native C++ MPI scheduler in Caffe binary (mpicaffe.cpp):
aprun -n 100 -d 16 $CAFFE_ROOT/build/tools/mpicaffe #TODO: input arguments...

4. monitor training:
python ./parse_logs.py


TODO: make the following settings easy to customize.

- mean_file location
- path to training LMDB
- path to val/testing LMDB
- [solver.prototxt...]
- CAFFE_ROOT in time_nets.sh (and probably in mpi_dnn.py)
- dnn_dsx/train_list.txt

