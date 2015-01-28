#ln -s ../solver_generator.py 
#ln -s ../random_net_defs.py 
mkdir gen1

for((seed=0; seed<3; seed=$seed+1))
do
  mkdir ./gen1/$seed
  #python ./random_net_generator.py --seed $seed --phase deploy > gen1/$seed/deploy.prototxt
  #python ./random_net_generator.py --seed $seed --phase trainval > gen1/$seed/trainval.prototxt
  python mutate_net.py --model=VGG_F/deploy.prototxt --model_out=gen1/$seed/deploy.prototxt --seed=$seed
  python mutate_net.py --model=VGG_F/trainval.prototxt --model_out=gen1/$seed/trainval.prototxt --seed=$seed
  python ../solver_generator.py gen1/$seed > gen1/$seed/solver.prototxt
done

