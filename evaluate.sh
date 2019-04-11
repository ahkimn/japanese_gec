DIR_DATA=input

DIR=input_sourced
EXT=.csv

mkdir -p temp

declare -a arr=("cnn_5" "cnn_7" "cnn_9" "lstm")

for MODELNAME in "${arr[@]}"; do

	DIR_SAVE=model/$MODELNAME/checkpoints

	for FILEPATH in $(find $DIR -type f -name "*$EXT"); do

		OUTPUT_SOURCE="$FILENAME.source"
		OUTPUT_TARGET="$FILENAME.target"

		python main.py process_csv $FILEPATH $OUTPUT_SOURCE $OUTPUT_TARGET

		# FILENAME="${FILEPATH##*/}"
		# FILENAME="${FILENAME%.*}"		

		# fairseq-preprocess --source-lang source --target-lang target --testpref temp/data --destdir temp \
		# --tgtdict $DIR_DATA/processed/dict.target.txt --srcdict $DIR_DATA/processed/dict.source.txt

		# touch temp/gen.out

		# OUTPUT_DIR=output_sourced/$MODELNAME/$FILENAME

		# mkdir -p $OUTPUT_DIR

		# touch $OUTPUT_DIR/gen.out.org
		# touch $OUTPUT_DIR/gen.out.ref
		# touch $OUTPUT_DIR/gen.out.sys

		# CUDA_VISIBLE_DEVICES=1 fairseq-generate --path $DIR_SAVE/checkpoint_best.pt --batch-size 64 --results-path output/tmp \
		# --memory-efficient-fp16 temp > temp/gen.out

		# grep ^S temp/gen.out > $OUTPUT_DIR/gen.out.org
		# grep ^T temp/gen.out > $OUTPUT_DIR/gen.out.ref
		# grep ^H temp/gen.out > $OUTPUT_DIR/gen.out.sys

	done
done

rm -r temp
