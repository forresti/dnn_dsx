


for((i=0; i<1000; i++))
do
    myDir=/nscratch/forresti/dsx_backup/nets_backup_1-9-15/$i

    python ./compute_throughput.py $myDir > results/${i}.throughput

done


