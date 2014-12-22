
#TODO: store results on sqlite3 database


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


if __name__ == "__main__":

    forward_time = get_forward_time(1)
    print forward_time    

