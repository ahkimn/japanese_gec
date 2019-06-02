DIR_DATA=input
DIR=input_sourced

COMMAND=$1

echo "COMMAND: $1"

if [ "$1" == "preprocess" ];
then

python mv_input.py

DIR_DATA=input

fairseq-preprocess --source-lang source --target-lang target \
	--trainpref $DIR_DATA/train --validpref $DIR_DATA/validation --testpref $DIR_DATA/test \
	--destdir $DIR_DATA/processed

# Example Usage:
# ./run.sh preprocess

elif [ "$1" == "generate" ];
then

	MODELNAME=$2
	FILEPATH=$3
	OUTPUT_FOLDER=$4

	echo "Architecture: $2"
	echo "Dataset: $3"
	echo "Ouput Directory: output/$OUTPUT_FOLDER"

	# Example Usage:
	# ./run.sh generate cnn_current student_diff test

	mkdir -p temp

	DIR_SAVE=model/$MODELNAME/checkpoints

	fairseq-preprocess --source-lang source --target-lang target --testpref input/$FILEPATH --destdir temp \
	--tgtdict $DIR_DATA/processed/dict.target.txt --srcdict $DIR_DATA/processed/dict.source.txt

	touch temp/gen.out

	OUTPUT_DIR=output/$OUTPUT_FOLDER
	mkdir -p $OUTPUT_DIR

	touch $OUTPUT_DIR/gen.out.org
	touch $OUTPUT_DIR/gen.out.ref
	touch $OUTPUT_DIR/gen.out.sys

	CUDA_VISIBLE_DEVICES=1 fairseq-generate --path $DIR_SAVE/checkpoint_best.pt \
											--batch-size 64 \
											--max-len-a 1 \
											--max-len-b 10 \
											--results-path output/tmp \
											--print-alignment \
											--memory-efficient-fp16 \
											temp > temp/gen.out

	grep ^S temp/gen.out > $OUTPUT_DIR/gen.out.org
	grep ^T temp/gen.out > $OUTPUT_DIR/gen.out.ref
	grep ^H temp/gen.out > $OUTPUT_DIR/gen.out.sys
	grep ^P temp/gen.out > $OUTPUT_DIR/gen.out.prob
	grep ^A temp/gen.out > $OUTPUT_DIR/gen.out.align

	# rm -r temp

elif [ "$1" == "process" ];
then

	# Example Usage:
	# ./run.sh process test test

	mkdir -p temp

	INPUT_FOLDER=$2
	OUTPUT_FOLDER=$3

	echo "Input Directory: output/$INPUT_FOLDER"
	echo "Output Directory: output/$OUTPUT_FOLDER"

	INPUT_DIR=output/$INPUT_FOLDER
	OUTPUT_DIR=output/$OUTPUT_FOLDER
	mkdir -p $OUTPUT_DIR

	python main.py sort_sentences $INPUT_DIR/gen.out.org temp/temp.org
	python main.py sort_sentences $INPUT_DIR/gen.out.ref temp/temp.ref
	python main.py sort_sentences $INPUT_DIR/gen.out.sys temp/temp.sys
	python main.py sort_sentences $INPUT_DIR/gen.out.prob temp/temp.prob
	python main.py sort_sentences $INPUT_DIR/gen.out.align temp/temp.align

	python main.py post_process temp/temp.org temp/temp.align temp/temp.sys temp/temp.prob temp/temp.sys_post_00 0.01
	python main.py post_process temp/temp.org temp/temp.align temp/temp.sys temp/temp.prob temp/temp.sys_post_0 0.02
	python main.py post_process temp/temp.org temp/temp.align temp/temp.sys temp/temp.prob temp/temp.sys_post_1 0.05
	python main.py post_process temp/temp.org temp/temp.align temp/temp.sys temp/temp.prob temp/temp.sys_post_2 0.1
	python main.py post_process temp/temp.org temp/temp.align temp/temp.sys temp/temp.prob temp/temp.sys_post_3 0.2
	
	mv temp/temp.org $OUTPUT_DIR/out.org
	mv temp/temp.ref $OUTPUT_DIR/out.ref
	mv temp/temp.sys $OUTPUT_DIR/out.sys
	mv temp/temp.prob $OUTPUT_DIR/out.prob
	mv temp/temp.align $OUTPUT_DIR/out.align
	
	mv temp/temp.sys_post_00 $OUTPUT_DIR/out.filter_0_01
	mv temp/temp.sys_post_0 $OUTPUT_DIR/out.filter_0_02
	mv temp/temp.sys_post_1 $OUTPUT_DIR/out.filter_0_05
	mv temp/temp.sys_post_2 $OUTPUT_DIR/out.filter_0_1
	mv temp/temp.sys_post_3 $OUTPUT_DIR/out.filter_0_2

    rm -r temp

elif [ "$1" == "filter_probabilities" ];
then

	INPUT_FOLDER=$2
	OUTPUT_FOLDER=$3
	echo "Input Directory: output/$INPUT_FOLDER"

	INPUT_DIR=output/$INPUT_FOLDER
	OUTPUT_DIR=output/$OUTPUT_FOLDER
	mkdir -p $OUTPUT_DIR

	python main.py filter_probabilities $INPUT_DIR/out.sys $INPUT_DIR/out.prob $INPUT_DIR/out.org $INPUT_DIR/out.ref $OUTPUT_DIR
fi
