DIR_DATA=input

declare -a arr=("$MODELNAME" "cnn_7" "cnn_9" "lstm")

for MODELNAME in "${arr[@]}"; do

	DIR_SAVE=model/$MODELNAME/checkpoints

	for i in {1..114}; do

		echo "Creating output from Rule: $i"

		TEST_DIR=$DIR_DATA/$i

		fairseq-preprocess --source-lang source --target-lang target --testpref $TEST_DIR/test --destdir output/tmp/$i \
			--tgtdict $DIR_DATA/processed/dict.target.txt --srcdict $DIR_DATA/processed/dict.source.txt

		touch output/tmp/$i/gen.out

		mkdir -p output/$MODELNAME/$i

		touch output/$MODELNAME/$i/gen.out.org
		touch output/$MODELNAME/$i/gen.out.ref
		touch output/$MODELNAME/$i/gen.out.sys

		CUDA_VISIBLE_DEVICES=1 fairseq-generate --path $DIR_SAVE/checkpoint_best.pt --batch-size 64 --results-path output/tmp \
			--memory-efficient-fp16 output/tmp/$i > output/tmp/$i/gen.out

		grep ^S output/tmp/$i/gen.out > output/$MODELNAME/$i/gen.out.org
		grep ^T output/tmp/$i/gen.out > output/$MODELNAME/$i/gen.out.ref
		grep ^H output/tmp/$i/gen.out > output/$MODELNAME/$i/gen.out.sys
done

