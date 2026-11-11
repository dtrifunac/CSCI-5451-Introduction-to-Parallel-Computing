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

# module load ompi/3.1.6/gnu-8.2.0
# module load ompi

# options to pass to mpirun
MPIOPTS=""
MPIOPTS+=" --mca mca_base_component_show_load_errors 0"

# No output, 500 simulation steps
OUTPUT=0
STEPS=500

# Full performance run, problem data for rod width and number of procs
ALLWIDTHS="6400 25600 102400 204800"
ALLNP="1 2 4 8 10 16 32 64 128" 

# # Smaller testing runs
# ALLWIDTHS="6400 25600"
# ALLNP="1 16 64 128" 

# Iterate over all proc/data file combos
for NP in $ALLNP; do 
    for WIDTH in $ALLWIDTHS; do
        echo HEAT $STEPS $WIDTH with $NP procs
        /usr/bin/time -f "runtime: procs $NP width $WIDTH realtime %e" \
                      mpirun $MPIOPTS -np $NP ./heat_mpi $STEPS $WIDTH $OUTPUT
        echo
    done
done
