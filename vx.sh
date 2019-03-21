# python mv_file.py

DIR_DATA=input

# fairseq-preprocess --source-lang source --target-lang target \
# 	--trainpref $DIR_DATA/train --validpref $DIR_DATA/validation --testpref $DIR_DATA/test \
# 	--destdir $DIR_DATA/processed

DIR_SAVE=model/test/checkpoints

mkdir -p $DIR_SAVE

# CUDA_VISIBLE_DEVICES=1 fairseq-train $DIR_DATA/processed --lr 0.12 --clip-norm 0.1 --dropout 0.1 --max-tokens 10000 \
    # --arch fconv_mini --save-dir $DIR_SAVE --batch-size 64  --memory-efficient-fp16 


# for i in {1..114}
# do
# 	echo "Creating output from Rule: $i"

# 	TEST_DIR=$DIR_DATA/$i

# 	fairseq-preprocess --source-lang source --target-lang target --testpref $TEST_DIR/test --destdir output/tmp/$i \
# 		--tgtdict $DIR_DATA/processed/dict.target.txt --srcdict $DIR_DATA/processed/dict.source.txt

# 	touch output/tmp/$i/gen.out
# 	touch output/tmp/$i/gen.out.org
# 	touch output/tmp/$i/gen.out.ref
# 	touch output/tmp/$i/gen.out.sys

# 	CUDA_VISIBLE_DEVICES=1 fairseq-generate --path $DIR_SAVE/checkpoint_best.pt --batch-size 64 --results-path output/tmp \
# 		--memory-efficient-fp16 output/tmp/$i > output/tmp/$i/gen.out

# 	grep ^S output/tmp/$i/gen.out | cut -f2- > output/tmp/$i/gen.out.org
# 	grep ^T output/tmp/$i/gen.out | cut -f2- > output/tmp/$i/gen.out.ref
# 	grep ^H output/tmp/$i/gen.out | cut -f3- > output/tmp/$i/gen.out.sys
# done
