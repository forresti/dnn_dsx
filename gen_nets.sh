mkdir nets

#seed=1
for((seed=125; seed<1000; seed=$seed+1))
do
  mkdir ./nets/$seed
  python ./random_net_generator.py --seed $seed --phase deploy > nets/$seed/deploy.prototxt
  python ./random_net_generator.py --seed $seed --phase trainval > nets/$seed/trainval.prototxt
  python ./solver_generator.py nets/$seed > nets/$seed/solver.prototxt
done

