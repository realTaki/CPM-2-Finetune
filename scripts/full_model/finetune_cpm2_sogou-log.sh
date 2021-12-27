#! /bin/bash

WORKING_DIR=/root/thu-plm/CPM-2-Finetune


NUM_WORKERS=2
GPUS_PER_NODE=

HOST_FILE="${WORKING_DIR}/configs/host_files/hostfile-cpm2"

MP_SIZE=4


DATA_EXT=".json"
DATA_PATH="/mnt/sfs_turbo/data/Sogou-log"

CONFIG_PATH="${WORKING_DIR}/configs/model/cpm2_config.json"
CKPT_PATH="/root/thu-plm/checkpoints/cpm2"


BATCH_SIZE=8
GRAD_ACC=8
LR=0.000005
TRAIN_ITER=20000
EPOCHS=10

SAVE_PATH="${WORKING_DIR}/results/cpm2_finetune_sogou-log"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_full_model.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_cn"

ENC_LEN=512
DEC_LEN=256


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-ext ${DATA_EXT}"
OPTS+=" --data-name sogou-log"
OPTS+=" --data-impl mmap"
OPTS+=" --lazy-loader"
OPTS+=" --tokenizer-type GPT2BPETokenizer"
OPTS+=" --split 949,50,1"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 400"
OPTS+=" --eval-interval 200"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 10"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do_train"
OPTS+=" --do_valid"
# OPTS+=" --do_eval"
# OPTS+=" --do_infer"
OPTS+=" --epochs ${EPOCHS}"


CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE} ${WORKING_DIR}/finetune_cpm2.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log

set +x
