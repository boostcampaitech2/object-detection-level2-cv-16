for i in 2 3 4 5 6 7 8 9
do

python testset_to_wandb.py ${i}

rm -r wandb

done