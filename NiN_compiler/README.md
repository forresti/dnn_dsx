
------
some more setup instructions...

(caffe_pb2.py gets compiled when you do `make pycaffe`)
#ln -s caffe/python/caffe/proto/caffe_pb2.py ./

-------

Getting compiled NiN to train

(uh, it doesn't train, loss is stuck at 6.9)

0. fixed dropout
-> still doesn't train

0. set conv/cccp layer `std` weight initialization to 0.01 and 0.05, as in original NiN
(previously, I had set them all to 0.01)
-> seems to train now

TODO
0. try `std: 0.05` for all conv/cccp layers
-> we get loss=56 instead of loss=6.9 at the beginning of training, and it goes to loss=87 after the first iteration.

0. try `std: 0.05` for all layers, except the final layer has `std: 0.01`
-> first, it says loss=13 after 0 iter, then loss=6.9 after 20 iter, (then hopefully loss goes down) 

0. try `std: 0.05 for all cccp, and `std: 0.01` for all conv 

