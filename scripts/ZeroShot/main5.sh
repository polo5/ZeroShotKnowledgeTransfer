#!/bin/sh

export CUDA_HOME=/opt/cuda-9.0.176.1/
source activate pytorch

EXECUTABLE_FILE=/afs/inf.ed.ac.uk/user/s17/s1771851/git/ZeroShotKnowledgeTransfer/main.py
LOG_DIR=/afs/inf.ed.ac.uk/user/s17/s1771851/logs
PRETRAINED_MODELS_DIR=/disk/scratch/s1771851/Pretrained/
DATASETS_DIR=/disk/scratch/s1771851/Datasets/Pytorch

python ${EXECUTABLE_FILE} \
--dataset CIFAR10 \
--total_n_pseudo_batches 8e4 \
--n_generator_iter 1 \
--n_student_iter 10 \
--batch_size 128 \
--z_dim 100 \
--student_learning_rate 2e-3 \
--generator_learning_rate 1e-3 \
--teacher_architecture WRN-40-2 \
--student_architecture WRN-40-1 \
--KL_temperature 1 \
--AT_beta 250 \
--pretrained_models_path ${PRETRAINED_MODELS_DIR} \
--datasets_path ${DATASETS_DIR} \
--log_directory_path ${LOG_DIR} \
--save_final_model True \
--save_n_checkpoints 0 \
--save_model_path ${LOG_DIR} \
--seeds 0 1 2 \
--workers 2 \
--use_gpu True