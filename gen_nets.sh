mkdir nets

#seed=1
for((seed=0; seed<50; seed=$seed+1))
do
  mkdir ./nets/$seed
  python ./random_net_generator.py --seed $seed --phase deploy > nets/$seed/deploy.prototxt
  python ./random_net_generator.py --seed $seed --phase trainval > nets/$seed/trainval.prototxt
done

