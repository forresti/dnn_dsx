import os

#recursive touch
def touch_r(path):
    os.system('find %s -exec touch -h {} \;' %path)

if __name__ == "__main__":
    touch_r('.')
    touch_r('$MEMBERWORK/csc103/dnn_exploration/')
