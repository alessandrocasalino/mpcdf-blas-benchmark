#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J octopus_gpu_benchmark
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="apu"   #   providing APUs.
#SBATCH --gres=gpu:1         # Request 2 APUs per node.
#SBATCH --ntasks-per-node=1  # Run one task per APU
#SBATCH --cpus-per-task=24   #   using 24 cores each.
#SBATCH --mail-type=NONE
#SBATCH --mail-user=alessandro.casalino@mpcdf.mpg.de
#SBATCH --time=0:05:00
#SBATCH -p apudev

module purge
module load rocm/6.4

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export LD_PRELOAD=/mpcdf/soft/RHEL_9/packages/x86_64/gcc/14.1.0/lib64/libstdc++.so.6

#srun rocprof-compute profile -n zprojector_bra_phase -k zprojector_bra_phase -- ../../install/bin/octopus
srun rocprofv3 --hip-trace --marker-trace --kernel-trace --memory-copy-trace --output-format pftrace -- ./build/octopus_devel

