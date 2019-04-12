DIR_DATA=input

DIR=input_sourced
EXT=.csv

mkdir -p temp

# python main.py remove_pairs input/teacher.source input/teacher.target input/teacher_same.source input/teacher_same.target 0
# python main.py remove_pairs input/student.source input/student.target input/student_same.source input/student_same.target 0

declare -a arch=("cnn_5" "cnn_7" "cnn_9" "lstm")
declare -a data=("teacher_same" "student_same")

for MODELNAME in "${arch[@]}"; do

	DIR_SAVE=model/$MODELNAME/checkpoints

	for FILEPATH in "${data[@]}"; do

		fairseq-preprocess --source-lang source --target-lang target --testpref input/$FILEPATH --destdir temp \
		--tgtdict $DIR_DATA/processed/dict.target.txt --srcdict $DIR_DATA/processed/dict.source.txt

		touch temp/gen.out

		OUTPUT_DIR=output/$MODELNAME/$FILEPATH

		mkdir -p $OUTPUT_DIR

		touch $OUTPUT_DIR/gen.out.org
		touch $OUTPUT_DIR/gen.out.ref
		touch $OUTPUT_DIR/gen.out.sys

		CUDA_VISIBLE_DEVICES=1 fairseq-generate --path $DIR_SAVE/checkpoint_best.pt --batch-size 64 --results-path output/tmp \
		--memory-efficient-fp16 temp > temp/gen.out

		grep ^S temp/gen.out > $OUTPUT_DIR/gen.out.org
		grep ^T temp/gen.out > $OUTPUT_DIR/gen.out.ref
		grep ^H temp/gen.out > $OUTPUT_DIR/gen.out.sys

	done
done








# declare -a arr=("cnn_5" "cnn_7" "cnn_9" "lstm")

# for MODELNAME in "${arr[@]}"; do

# 	DIR_SAVE=model/$MODELNAME/checkpoints

# 	for FILEPATH in $(find $DIR -type f -name "*$EXT"); do

# 		OUTPUT_SOURCE="$FILENAME.source"
# 		OUTPUT_TARGET="$FILENAME.target"

# 		python main.py process_csv $FILEPATH $OUTPUT_SOURCE $OUTPUT_TARGET

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

# 	done
# done

rm -r temp
