#!/bin/sh

DIM=(64 128 256)
perfdir="perfs"

for d in "${DIM[@]}"
do
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_DGL_cora_${d} train.py --n-epochs 100 --dataset cora --n-hidden ${d} >> log_DGL.txt
	python3 gen_stats.py ${perfdir}/perf_DGL_cora_${d} >> ${perfdir}/perf_DGL_cora_${d}_stats
 done
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_FusedMM_cora_${d} train.py --n-epochs 100 --dataset cora --n-hidden ${d} --gcn2 >> log_FusedMM.txt
	python3 gen_stats.py ${perfdir}/perf_FusedMM_cora_${d} >> ${perfdir}/perf_FusedMM_cora_${d}_stats 
done
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_DGL_citeseer_${d} train.py --n-epochs 100 --dataset citeseer --n-hidden ${d} >> log_DGL.txt
	python3 gen_stats.py ${perfdir}/perf_DGL_citeseer_${d} >> ${perfdir}/perf_DGL_citeseer_${d}_stats 
done
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_FusedMM_citeseer_${d} train.py --n-epochs 100 --dataset citeseer --n-hidden ${d} --gcn2 >> log_FusedMM.txt
	python3 gen_stats.py ${perfdir}/perf_FusedMM_citeseer_${d} >> ${perfdir}/perf_FusedMM_citeseer_${d}_stats 
done
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_DGL_pubmed_${d} train.py --n-epochs 100 --dataset pubmed --n-hidden ${d} >> log_DGL.txt
	python3 gen_stats.py ${perfdir}/perf_DGL_pubmed_${d} >> ${perfdir}/perf_DGL_pubmed_${d}_stats
 done
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_FusedMM_pubmed_${d} train.py --n-epochs 100 --dataset pubmed --n-hidden ${d} --gcn2 >> log_FusedMM.txt
	python3 gen_stats.py ${perfdir}/perf_FusedMM_pubmed_${d} >> ${perfdir}/perf_FusedMM_pubmed_${d}_stats 
done
done
