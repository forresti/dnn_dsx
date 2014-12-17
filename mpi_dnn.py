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
    READY=100 #worker is ready
    DATA=200 #master has new homework for a worker
    DONE=300 #job queue is complete 

MASTER=0 #master is rank 0

#master/slave scheduler for Caffe training
# TODO: extend this so that workers can iterate over multiple jobs.
def scheduler(caffe_job_list):
    comm = MPI.COMM_WORLD
    print "Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size)
    rank = comm.rank
    num_workers = comm.size-1
  
    if rank > 0: #worker
        #while True:
        comm.send(obj={}, dest=MASTER, tag=tags.READY)

        status = MPI.Status() 
        jobinfo = comm.recv(source=MASTER, tag=MPI.ANY_TAG, status=status) #TODO: check the tag (DATA or DONE) 
        print "I'm rank %d"%rank, "my next task is: ", jobinfo 

    elif rank == 0: #master
        #while True: #if we Ctrl+C this, future MPI jobs seem to hang. (why?)
        #TODO: perhaps iterate over length of Caffe job list

        for i in xrange(0, comm.size-1): 
            status = MPI.Status()
 
            comm.recv(obj=None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) #blocking receive
            workerID = status.Get_source()
            assert status.Get_tag() == tags.READY #worker only contacts master when it's ready for more work.
            print "in master: got message from rank=%d, tag=%s" %(workerID, status.Get_tag())

            comm.send(obj={'caffe_job_id':caffe_job_list[i]}, dest=workerID, tag=tags.DATA)

        '''
        #simple point-to-point w/o synchronization:
        for workerID in xrange(1, comm.size):
            #comm.send(obj={'caffe_job_id':caffe_job_list[workerID]}, dest=workerID, tag=tags.DATA)
            comm.send(obj={}, dest=workerID, tag=tags.DATA)
        '''

    comm.Barrier()   # wait for everybody to synchronize _here_

def get_work_list():
    caffe_job_list = [1,2,3,4] #TODO: read this from directory structure
    return caffe_job_list

if __name__ == "__main__":
    caffe_job_list = get_work_list()
    scheduler(caffe_job_list)

