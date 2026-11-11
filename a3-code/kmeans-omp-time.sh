#!/bin/bash -l

# Fixed parameters for all runs of kmeans
DATADIR="mnist-data"
NCLUST=20
MAXITERS=500
mkdir -p outdirs   # subdir or all output kmeans output

# Full performance benchmark of all combinations of data files and
# processor counts
ALLDATA="digits_all_5e3.txt digits_all_1e4.txt digits_all_3e4.txt"
ALLNP="10 13 16 32"

# ALLDATA="digits_all_3e4.txt"
# ALLNP="8"

# # Small sizes for testing
# ALLDATA="digits_all_3e4.txt"
# ALLNP="1 2 4 8"

# Iterate over all proc/data file combos
for NP in $ALLNP; do 
    for DATA in $ALLDATA; do
        echo KMEANS $DATA with $NP procs
        OUTDIR=outdirs/outdir_${DATA}_${NP}
        export OMP_NUM_THREADS=$NP
        /usr/bin/time -f "runtime: procs $NP data $DATA realtime %e" \
                     ./kmeans_omp $DATADIR/$DATA $NCLUST $OUTDIR $MAXITERS
        echo
    done
done
