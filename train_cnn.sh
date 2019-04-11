DIR_DATA=input
DIR_SAVE=model/cnn_5/checkpoints

mkdir -p $DIR_SAVE

CUDA_VISIBLE_DEVICES=1 fairseq-train $DIR_DATA/processed --lr 0.15 --clip-norm 0.1 --dropout 0.1 --max-tokens 10000 \
    --arch fconv_jp_5 --save-dir $DIR_SAVE --batch-size 48  --memory-efficient-fp16 

DIR_SAVE=model/cnn_7/checkpoints

mkdir -p $DIR_SAVE

CUDA_VISIBLE_DEVICES=1 fairseq-train $DIR_DATA/processed --lr 0.15 --clip-norm 0.1 --dropout 0.1 --max-tokens 10000 \
    --arch fconv_jp_7 --save-dir $DIR_SAVE --batch-size 48  --memory-efficient-fp16 

DIR_SAVE=model/cnn_9/checkpoints

mkdir -p $DIR_SAVE

CUDA_VISIBLE_DEVICES=1 fairseq-train $DIR_DATA/processed --lr 0.15 --clip-norm 0.1 --dropout 0.1 --max-tokens 10000 \
    --arch fconv_jp_9 --save-dir $DIR_SAVE --batch-size 48  --memory-efficient-fp16 

