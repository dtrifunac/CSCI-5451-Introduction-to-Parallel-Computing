#!/bin/bash -l
# basic job stats
#SBATCH --time=0:30:00
#SBATCH --ntasks=128
#SBATCH --mem=8g
#SBATCH --partition small

# set output based on script name
#SBATCH --output=%x.job-%j.out
#SBATCH --error=%x.job-%j.out

# charge compute time to course group account
#SBATCH --account csci5451

# # enable email notification of job completion
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=kauffman@umn.edu

# ADJUST: location of executable
cd ~/mesabi-a2-5451/

# set path to MPI, show path to mpirun, helps with versioning and
# debugging in output
module load ompi/4.0.0/gnu-8.2.0-centos7
which mpirun

MPIOPTS=""
MPIOPTS+=" --mca mca_base_component_show_load_errors 0"

# Fixed parameters for all runs of kmeans
DATADIR="mnist-data"
NCLUST=20
MAXITERS=500
mkdir -p outdirs   # subdir or all output kmeans output

# Full performance benchmark of all combinations of data files and
# processor counts
ALLDATA="digits_all_5e3.txt digits_all_1e4.txt digits_all_3e4.txt"
ALLNP="1 2 4 8 10 16 32 64 128"

# # Small sizes for testing
# ALLDATA="digits_all_3e4.txt"
# ALLNP="1 16 128"

# Iterate over all proc/data file combos
for NP in $ALLNP; do 
    for DATA in $ALLDATA; do
        echo KMEANS $DATA with $NP procs
        OUTDIR=outdirs/outdir_${DATA}_${NP}
        /usr/bin/time -f "runtime: procs $NP data $DATA realtime %e" \
                      mpirun $MPIOPTS -np $NP ./kmeans_mpi $DATADIR/$DATA $NCLUST $OUTDIR $MAXITERS
        echo
    done
done
