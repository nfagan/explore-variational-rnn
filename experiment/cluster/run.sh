#!/bin/bash
#SBATCH --job-name=vib_replicate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=24G

module purge
module load anaconda3/2024.02
source /home/naf264/.bashrc
# conda env create -f environment.yml -p /scratch/naf264/conda/pn0
conda activate /scratch/naf264/conda/pn0

cd /home/naf264/source/explore-variational-rnn/experiment
python vib_replicate.py