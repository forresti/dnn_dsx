
mkdir nets_backup

#for d in ./nets/*
for((i=0; i<1000; i++))
do
  d=./nets/$i
  output=./nets_backup/$i
  mkdir $output

  newestState=`ls -t $d/*solverstate | head -1` #thx: stackoverflow.com/questions/5885934
  newestModel=`ls -t $d/*caffemodel | head -1` 

  echo $newestState

  touch $newestState
  touch $newestModel
  touch $d/*prototxt

  cp $newestState $output
  cp $newestModel $output
  cp $d/*.log* $output
  cp $d/*prototxt $output
done

