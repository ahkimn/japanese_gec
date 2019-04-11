DIR_DATA=input
DIR_SAVE=model/lstm/checkpoints

mkdir -p $DIR_SAVE

CUDA_VISIBLE_DEVICES=1 fairseq-train $DIR_DATA/processed --lr 0.15 --clip-norm 0.1 --dropout 0.1 --max-tokens 10000 \
    --arch lstm --save-dir $DIR_SAVE --batch-size 64  --memory-efficient-fp16 

