#!/usr/bin/env python
from mpi4py import MPI
'''
MPI-based scheduler of DNN training in Caffe.

#usage:
aprun -n 4 -d 1 python ./mpi_dnn.py #small-scale ... 4 processes, 1 node
aprun -n 100 -d 16 python ./mpi_dnn.py #large-scale ... 100 processes, 100 nodes

'''
#tags = Enum('READY', 'DATA', 'DONE') #thx: stackoverflow.com/questions/21088420

#poor man's enum (without installing enum library)
class tags:
    # TODO: perhaps add a tag for 'CAFFE_FAILED' or something like that (for master to send to worker)
    READY=0
    DATA=1
    DONE=2

MASTER=0 #master is rank 0

#master/slave scheduler for Caffe training
# (for now, at least,) you need to Ctrl+C this when done; master process isn't smart enough to exit.
def scheduler(caffe_job_list):
    comm = MPI.COMM_WORLD
    print "Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size)

    rank = comm.rank
    '''
    if rank > 0: #slave
    
        #while True:
        
        comm.send(obj={}, MASTER, tags.DATA) 

        #comm.recv(...)

    elif rank == 0: #master
        #while True: #keeps printing the same thing over and over. why doesn't it block?
        for i in xrange(0,10):
            status = MPI.Status() 
            comm.recv(obj=None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) #blocking receive
            print "in master: got message from rank=%d, tag=%s" %(status.Get_source(), status.Get_tag())
    '''
    comm.Barrier()   # wait for everybody to synchronize _here_

def get_work_list():
    caffe_job_list = [1,2,3,4] #TODO: read this from directory structure
    return caffe_job_list

if __name__ == "__main__":
    caffe_job_list = get_work_list()
    scheduler(caffe_job_list)

