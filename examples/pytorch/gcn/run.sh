#!/bin/sh

DIM=(32 64 128)

for d in "${DIM[@]}"
do
 for i in 1 2 3 4 5
 do
	python3 train.py --n-epochs 100 --dataset cora --n-hidden ${d} >> log_DGL.txt
 done
 for i in 1 2 3 4 5
 do
	python3 train.py --n-epochs 100 --dataset cora --n-hidden ${d} --gcn2 >> log_FusedMM.txt
 done
 for i in 1 2 3 4 5
 do
	python3 train.py --n-epochs 100 --dataset citeseer --n-hidden ${d} >> log_DGL.txt
 done
 for i in 1 2 3 4 5
 do
	python3 train.py --n-epochs 100 --dataset citeseer --n-hidden ${d} --gcn2 >> log_FusedMM.txt
 done
 for i in 1 2 3 4 5
 do
	python3 train.py --n-epochs 100 --dataset pubmed --n-hidden ${d} >> log_DGL.txt
 done
 for i in 1 2 3 4 5
 do
	python3 train.py --n-epochs 100 --dataset pubmed --n-hidden ${d} --gcn2 >> log_FusedMM.txt
 done
done
