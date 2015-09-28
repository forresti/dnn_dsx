import os
from sys import argv
from pprint import pprint
from IPython import embed

#@param files = list of '.caffemodel' or '.snapshot' files
#return the list, without the latest model (sorted by iteration number)
def pop_latest(files):
    #filename format: blahblah_iter_<iteration count>.caffemodel
    files.sort(key=lambda f: int(f.split('.')[0].split('_')[-1]))
    #embed()
    files.pop() #remove last item
    return files

def rm_list(files):
    for f in files:
        os.remove(f)

def rm_old_caffemodels(net_dir):
    files = os.listdir(net_dir)
    caffemodels = [f for f in files if '.caffemodel' in f]
    snapshots = [f for f in files if '.solverstate' in f]

    caffemodels = pop_latest(caffemodels)
    snapshots = pop_latest(snapshots)

    caffemodels = [net_dir + '/' + c for c in caffemodels] 
    snapshots = [net_dir + '/' + c for c in snapshots]

    rm_list(caffemodels)
    rm_list(snapshots)

#net_dir = directory that contains directories of training runs
def rm_old_caffemodels_hierarchy(net_dir):
    for item in os.listdir(net_dir):
        fullpath=os.path.join(net_dir, item)
        if os.path.isdir(fullpath):
            print fullpath
            try:
                rm_old_caffemodels(fullpath)
            except:
                pass

#sample usage: python rm_old_caffemodels.py nets/vanilla_alexnet_drop0.8/
#sample usage: python rm_old_caffemodels.py /nscratch/forresti/_caffe_allreduce/nets
if __name__ == "__main__":
    if len(argv) < 2:
        print "Error: requires one command-line argument: directory of images to pad to square"
        sys.exit()
    
    net_dir = argv[1]

    #if net_dir directly contains caffemodels...
    #rm_old_caffemodels(net_dir)
    rm_old_caffemodels_hierarchy(net_dir)
    

