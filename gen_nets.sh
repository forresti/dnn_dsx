mkdir nets

#seed=1
for((seed=0; seed<50; seed=$seed+1))
do
  python ./random_net_generator.py $seed > nets/deploy$seed.txt
done

