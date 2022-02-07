#!/bin/sh

DIM=(64 128 256)
perfdir="perfs"

for d in "${DIM[@]}"
do
  dname="pubmed"
  for i in 1 2 3 4 5
  do
 	python3 -m cProfile -o ${perfdir}/perf_DGL_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} >> log_DGL.txt
 	python3 gen_stats.py ${perfdir}/perf_DGL_${dname}_${d} >> ${perfdir}/perf_DGL_${dname}_${d}_stats
  done
  for i in 1 2 3 4 5
  do
	python3 -m cProfile -o ${perfdir}/perf_FusedMM_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --gcn2 >> log_FusedMM.txt
	python3 gen_stats.py ${perfdir}/perf_FusedMM_${dname}_${d} >> ${perfdir}/perf_FusedMM_${dname}_${d}_stats 
 done
 dname="PUBMED"
 for i in 1 2 3 4 5
 do
 	python3 -m cProfile -o ${perfdir}/perf_DGL_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} >> log_DGL.txt
 	python3 gen_stats.py ${perfdir}/perf_DGL_${dname}_${d} >> ${perfdir}/perf_DGL_${dname}_${d}_stats 
 done
for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_FusedMM_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --gcn2 >> log_FusedMM.txt
	python3 gen_stats.py ${perfdir}/perf_FusedMM_${dname}_${d} >> ${perfdir}/perf_FusedMM_${dname}_${d}_stats 
 done
 dname="amazon"
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_DGL_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} >> log_DGL.txt
	python3 gen_stats.py ${perfdir}/perf_DGL_${dname}_${d} >> ${perfdir}/perf_DGL_${dname}_${d}_stats
 done
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_FusedMM_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --gcn2 >> log_FusedMM.txt
	python3 gen_stats.py ${perfdir}/perf_FusedMM_${dname}_${d} >> ${perfdir}/perf_FusedMM_${dname}_${d}_stats 
 done
 dname="coauthorp"
 for i in 1 2 3 4 5
 do
        python3 -m cProfile -o ${perfdir}/perf_DGL_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} >> log_DGL.txt
        python3 gen_stats.py ${perfdir}/perf_DGL_${dname}_${d} >> ${perfdir}/perf_DGL_${dname}_${d}_stats
 done
 for i in 1 2 3 4 5
 do
        python3 -m cProfile -o ${perfdir}/perf_FusedMM_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --gcn2 >> log_FusedMM.txt
        python3 gen_stats.py ${perfdir}/perf_FusedMM_${dname}_${d} >> ${perfdir}/perf_FusedMM_${dname}_${d}_stats
 done
 dname="reddit"
 for i in 1 2 3 4 5
 do
        python3 -m cProfile -o ${perfdir}/perf_DGL_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} >> log_DGL.txt
        python3 gen_stats.py ${perfdir}/perf_DGL_${dname}_${d} >> ${perfdir}/perf_DGL_${dname}_${d}_stats
 done
 for i in 1 2 3 4 5
 do
        python3 -m cProfile -o ${perfdir}/perf_FusedMM_${dname}_${d} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --gcn2 >> log_FusedMM.txt
        python3 gen_stats.py ${perfdir}/perf_FusedMM_${dname}_${d} >> ${perfdir}/perf_FusedMM_${dname}_${d}_stats
 done
done
