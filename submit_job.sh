#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=14000
#SBATCH --job-name=train_non_linear_pirl_on_stl
#SBATCH --mail-type=END
#SBATCH --mail-user=ab8700@nyu.edu
#SBATCH --output=slurm_%j.out

python pirl_stl_train_test.py --model-type 'res18' --lr 0.1 --tmax-for-cos-decay 50 --warm-start True  --only-train True --non-linear-head True --cont-epoch 101 --experiment-name e8_stl_pirl

