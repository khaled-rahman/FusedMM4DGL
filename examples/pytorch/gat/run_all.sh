#!/bin/sh

DIM=(64 128)
TSAMPLE=(4000)
perfdir="perfs"

for ts in "${TSAMPLE[@]}"
do
for d in "${DIM[@]}"
do
 dname="ogbn-protein"
 for i in 1 2 3 4 5
 do
 	python3 -m cProfile -o ${perfdir}/perf_DGL_${dname}_${d}_${ts} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --tsamples ${ts} >> log_DGL.txt
 	python3 gen_stats.py ${perfdir}/perf_DGL_${dname}_${d}_${ts}  >> ${perfdir}/perf_DGL_${dname}_${d}_${ts}_stats 
 done
for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_FusedMM_${dname}_${d}_${ts} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --gcn2 --tsamples ${ts} >> log_FusedMM.txt
	python3 gen_stats.py ${perfdir}/perf_FusedMM_${dname}_${d}_${ts} >> ${perfdir}/perf_FusedMM_${dname}_${d}_${ts}_stats 
 done
done
done
'''
 dname="amazon"
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_DGL_${dname}_${d}_${ts} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --tsamples ${ts} >> log_DGL.txt
	python3 gen_stats.py ${perfdir}/perf_DGL_${dname}_${d}_${ts} >> ${perfdir}/perf_DGL_${dname}_${d}_${ts}_stats
 done
 for i in 1 2 3 4 5
 do
	python3 -m cProfile -o ${perfdir}/perf_FusedMM_${dname}_${d}_${ts} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --gcn2 --tsamples ${ts} >> log_FusedMM.txt
	python3 gen_stats.py ${perfdir}/perf_FusedMM_${dname}_${d}_${ts} >> ${perfdir}/perf_FusedMM_${dname}_${d}_${ts}_stats 
 done
 dname="reddit"
 for i in 1 2 3 4 5
 do
        python3 -m cProfile -o ${perfdir}/perf_DGL_${dname}_${d}_${ts} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --tsamples ${ts} >> log_DGL.txt
        python3 gen_stats.py ${perfdir}/perf_DGL_${dname}_${d}_${ts} >> ${perfdir}/perf_DGL_${dname}_${d}_${ts}_stats
 done
 for i in 1 2 3 4 5
 do
        python3 -m cProfile -o ${perfdir}/perf_FusedMM_${dname}_${d}_${ts} train.py --n-epochs 100 --dataset ${dname} --n-hidden ${d} --gcn2 --tsamples ${ts} >> log_FusedMM.txt
        python3 gen_stats.py ${perfdir}/perf_FusedMM_${dname}_${d}_${ts} >> ${perfdir}/perf_FusedMM_${dname}_${d}_${ts}_stats
 done
done
done
'''
