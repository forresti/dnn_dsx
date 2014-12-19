
rough workflow:

1. generate DNNs
./gen_nets.sh
    calls random_net_generator.py
    outputs to [some folder]

2. time DNNs
./time_net.sh
    copies nets that are sufficiently fast to [some folder]

3. train DNNs (many concurrent training runs on compute cluster)
python ./mpi_dnn.py


TODO: make the following settings easy to customize.

- mean_file location
- path to training LMDB
- path to val/testing LMDB
- [solver.prototxt...]
- CAFFE_ROOT in time_nets.sh (and probably in mpi_dnn.py)

