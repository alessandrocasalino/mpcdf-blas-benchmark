#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J octopus_gpu_bench
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"    #    providing GPUs.
#SBATCH --gres=gpu:a100:1    # Request 4 GPUs per node.
#SBATCH --ntasks-per-node=1  # Run one task per GPU
#SBATCH --cpus-per-task=18   #   using 18 cores each.
#SBATCH --mem=125000
#SBATCH --mail-type=NONE
#SBATCH --mail-user=alessandro.casalino@mpcdf.mpg.de
#SBATCH --time=0:05:00
#SBATCH -p gpudev

module purge
module load cuda/12.6-nvhpcsdk_25

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

compute-sanitizer ./build/octopus_devel

