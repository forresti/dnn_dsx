import os
from time import sleep

#recursive touch
def touch_r(path):
  os.system('touch %s/*' %path)
  os.system('find %s -exec touch -h {} \;' %path) 

if __name__ == "__main__":

  while 1:
    touch_r('.')
    touch_r('$MEMBERWORK/csc103/dnn_exploration/') 
    sleep(60*60*24*7) #sleep 1 week, then do it again
    


