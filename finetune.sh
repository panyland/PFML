#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=pfml_finetune
#SBATCH --output=finetune_%j.log

cd /home/rqb592/dippa/PFML
source /home/rqb592/dippa/dippaenv/bin/activate
python finetune_pfml_pretrained_imu_models.py
