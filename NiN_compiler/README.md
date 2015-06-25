
Step 0: create the 1st generation by mutating your initial net.

for each generation...

Step 1: crossover
#crossover uses the 'seed' param to randomly choose a crossover point.
python crossover_net.py --parent0=gen1/seed/trainval.prototxt --parent1=gen1/seed/trainval.prototxt --model_out=.../gen2_crossover/trainval.prototxt --seed=? 

Step 2: mutation
python mutate_net.py --model=gen2_crossover/seed/trainval.prototxt --model_out=gen2/seed/trainval.prototxt --seed=?


TODO: make 'crossover' and 'mutation' also preserve some pre-trained weights from parent models.


------
some more setup instructions...

(caffe_pb2.py gets compiled when you do `make pycaffe`)
#ln -s caffe/python/caffe/proto/caffe_pb2.py ./


