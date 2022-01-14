#!/bin/sh
#BSUB -q gpua100
#BSUB -J "ViT"
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=8GB]"
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gpu-%J.out
#BSUB -eo gpu-%J.err
# -- end of LSF options --

# Load the cuda module
module load cuda/10.2
module load cudnn/v8.2.2.26-prod-cuda-10.2

export PYTHONPATH=~/miniconda3/envs/ml_ops/bin/:$PYTHONPATH
export PATH=~/miniconda3/envs/ml_ops/bin/:$PATH

wandb agent fill