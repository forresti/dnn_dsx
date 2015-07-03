
------
some more setup instructions...

(caffe_pb2.py gets compiled when you do `make pycaffe`)
#ln -s caffe/python/caffe/proto/caffe_pb2.py ./

-------

Getting compiled NiN to train

(uh, it doesn't train, loss is stuck at 6.9)

0. fixed dropout
(still doesn't train)

0. set conv/cccp layer `std` weight initialization to 0.01 and 0.05, as in original NiN
(previously, I had set them all to 0.01)
(seems to train now)

TODO
0. try `std: 0.05` for all conv/cccp layers

0. try `std: 0.05 for all cccp, and `std: 0.01` for all conv 

