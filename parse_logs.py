from pprint import pprint
import os
import time
#TODO: store results on sqlite3 database

update_trainlist = True

#@param netID = random seed for this net (e.g. 3 -> ./nets/3)
def get_forward_time(netID):
    dir = './nets/%d' %netID
    f = open(dir + '/timing.log')

    time_str = None
    line = f.readline()
    while line:
        if "Forward pass: " in line:
            time_str = line #I1220 20:47:35.469178  2823 caffe.cpp:247] Forward pass: 1940.47 milliseconds.
        line = f.readline()
    f.close()
    if time_str is None:
        return None
    time_substr = time_str.split("Forward pass: ")[1] #1940.47 milliseconds.
    time_float = float(time_substr.split(' ')[0]) #1940.47 
    return time_float


#get the filename of the latest training log, e.g. "train_Sun_2014_12_21__16_01_39.log"
def get_latest_log(netID):
    logdir = './nets/%d' %netID
    allfiles = os.listdir(logdir)

    files_with_time = []

    for f in allfiles:
        if not (f.startswith('train_') and f.endswith('.log')):
            continue
        timeOnly = f[len('train_'):]
        timeOnly = timeOnly[:-len('.log')]
        timeOnly = time.strptime(timeOnly, '%a_%Y_%m_%d__%H_%M_%S')
        files_with_time.append({'filename':f, 'time':timeOnly})

    files_with_time = sorted(files_with_time, key=lambda f:f['time']) #sort from old to new
    #pprint(files_with_time)
    return logdir + '/' + files_with_time[-1]['filename']

#@param log_filename: e.g. './nets/0/train_Sun_2014_12_21__16_01_39.log'
#@return num iter and accuracy
def get_current_accuracy(log_filename):
    '''
    look latest one of these:
    I1221 22:31:42.642572 23298 solver.cpp:247] Iteration 34000, Testing net (#0)
    I1221 22:34:36.420714 23298 solver.cpp:298]     Test net output #0: accuracy = 0.11198
    '''

    f = open(log_filename)
    test_results = [] #results for every time we test the net

    line = f.readline()
    while line:
        if "Testing net" in line:
            iter_str = line #...solver.cpp:247] Iteration 34000, Testing net (#0)
            accuracy_str = f.readline() #...solver.cpp:298]     Test net output #0: accuracy = 0.11198

            iter = iter_str.split("Iteration ")[1].split(',')[0]
            iter = int(iter)
            
            accuracy = accuracy_str.split("Test net output #0: ")[1].split('= ')[1]
            accuracy = float(accuracy)
            test_results.append({'accuracy':accuracy, 'iter':iter})

        if "error" in line:
            return "error"

        line = f.readline()

    #pprint(test_results)
    #test_results is already sorted, since we read the log file in order
    return test_results[-1]
    

def quick_test():
    forward_time = get_forward_time(1)
    #print forward_time    

    latest_log = get_latest_log(1)
    #print latest_log

    accuracy_dict = get_current_accuracy('./nets/0/train_Sun_2014_12_21__16_01_39.log')

def run_analytics():
    if update_trainlist:
        tl = open('train_list_.txt', 'w')

    for i in xrange(0,275):
        try:
            forward_time = get_forward_time(i)
            latest_log = get_latest_log(i)
            accuracy_dict = get_current_accuracy(latest_log)

            if accuracy_dict is "error":
                print "error in net: ",str(i)
                continue

            if accuracy_dict['accuracy'] > 0.2:
                print ' seed=%d, forward_time = %f ms, accuracy = %f at iter %d'%(i, forward_time, accuracy_dict['accuracy'], accuracy_dict['iter'])

            #if we're not at 10% accuracy by 50k iterations, prune this net. else, carry on training.
            if update_trainlist and (accuracy_dict['accuracy'] > 0.1 or accuracy_dict['iter']<50000): 
                tl.write('./nets/' + str(i) + '\n')

        except:
            #print "problem parsing seed=%d results"%i
            continue

    #throw some fresh nets into the hopper (total of 313 nets)
    #for i in xrange(275, 275+88):
    '''
    for i in xrange(275, 275+150):
        tl.write('./nets/' + str(i) + '\n')
    '''

    if update_trainlist:
        tl.close()

if __name__ == "__main__":


    #quick_test()
    run_analytics()

