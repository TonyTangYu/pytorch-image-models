#!/usr/bin/env bash

export PYTHONPATH=/mnt/lustre/share/pymc/py3:$PYTHONPATH
set -x

export PYTHONPATH=/mnt/petrelfs/tangyu/tso-project/pytorch-image-models:$PYTHONPATH

export NCCL_IB_TC=106
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0
export MASTER_ADDR=localhost
export MASTER_PORT=5678


export OMPI_MCA_btl_tcp_if_include="eth2" 

PARTITION=$1
JOB_NAME=$2
GPUS=$3
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

# export PROFILE=1
# GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 MC_COUNT_DISP=1000000 \
# OMPI_MCA_mpi_warn_on_fork=0 PYTHONWARNINGS=ignore \
srun -p ${PARTITION}  \
    --quotatype=spot   \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train_slurm.py \
    /mnt/petrelfs/share/images \
    --val-split val \
    -b 96 \
    --model resnet50 \
    --seed 84 \
    --log-interval 200 \
    --opt adam  \
    --momentum 0.8929465429717223 \
    --weight-decay  0.00013780586156178121 \
    --lr 26.521263439653904 \
    --epochs 90 \
    -vb 2   \
    --warmup-epochs 5 \
    --warmup-lr 0.01 \
    --channels-last     \
    --experiment resnet50 \
    2>&1 | tee ${JOB_NAME}.log

# SH-IDC1-10-142-5-[1-12,14-20,22,24-40,42-44,46-62,64,66-76,78,80,82-85,88-129,131-138,139]