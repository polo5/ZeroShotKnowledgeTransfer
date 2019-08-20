#!/bin/sh

export CUDA_HOME=/opt/cuda-9.0.176.1/
source activate pytorch

EXECUTABLE_FILE=/afs/inf.ed.ac.uk/user/s17/s1771851/git/ZeroShotKnowledgeTransfer/utils/transition_curves.py
LOG_DIR=/afs/inf.ed.ac.uk/user/s17/s1771851/logs
DATASETS_DIR=/disk/scratch/s1771851/Datasets/Pytorch
NETA_PATH=/disk/scratch/s1771851/Pretrained/Distillation/CIFAR10/WRN-40-2_to_WRN-16-1/last.pth.tar
NETB_PATH=/disk/scratch/s1771851/Pretrained/CIFAR10/WRN-40-2/last.pth.tar

python ${EXECUTABLE_FILE} \
--dataset CIFAR10 \
--netA_path ${NETA_PATH} \
--netB_path ${NETB_PATH} \
--netA_architecture WRN-16-1 \
--netB_architecture WRN-40-2 \
--check_test_accuracies True \
--use_train_set False \
--n_matching_images 1000 \
--try_load_indices False \
--n_adversarial_steps 200 \
--learning_rate 1 \
--datasets_path ${DATASETS_DIR} \
--log_directory_path ${LOG_DIR} \
--save_model_path ${LOG_DIR} \
--seed 0 \
--use_gpu True