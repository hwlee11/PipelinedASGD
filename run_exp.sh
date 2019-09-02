#/bin/bash

batch_list='16 32 64'
tau_list='4 8 16 32 64 128 256'
for j in $batch_list;do
for i in $tau_list;do

	python3.6 bp_pip_mnist_test_ver3.4_20190523_sendasync_dataloader.py --dataset cifar10 --manualSeed 2 --batch_size $j --epoch 220  --loss_save 'batch'$j'_tau'$i'_test.txt'
done
done
