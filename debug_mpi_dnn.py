#!/usr/bin/env python
from mpi4py import MPI
import os
import subprocess
'''
MPI-based scheduler of DNN training in Caffe.

#usage:
aprun -n 4 -d 1 python ./debug_mpi_dnn.py #small-scale ... 4 processes, 1 node

'''

CAFFE_ROOT = '/lustre/atlas/scratch/forresti/csc103/dnn_exploration/caffe-bvlc-master/'

#each MPI rank gets exactly one task. 
def scheduler_static():
    comm = MPI.COMM_WORLD
    print "Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size)

    #note that rank 0 hangs if it has to call caffe.

    if comm.rank > 0:
        #this works:
        #subprocess.call('ls', shell=True)

        #caffe hello world
        #cmd = CAFFE_ROOT + '/build/tools/caffe > rank %d.log 2>&1'%comm.rank
        cmd = CAFFE_ROOT + '.build_release/tools/caffe.bin 2>&1'
        #cmd = '/lustre/atlas/scratch/forresti/csc103/dnn_exploration/a.out' #fibonacci.c -- this works.
        print cmd
        subprocess.call(cmd, shell=True)
        #this command seems to do nothing. zero printouts, no stderr or stdout. WTF?

        print 'rank %d is done running caffe hello world' %comm.rank


    comm.Barrier()   # wait for everybody to synchronize _here_

if __name__ == "__main__":
    scheduler_static()

