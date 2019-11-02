SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
MODEL_DIR='model'

PREFIX='fairseq'

FUNCTION=$1
echo $FUNCTION

preprocess(){

	python mv_input.py $1;

	PROCESS_DIR="$MODEL_DIR/$PREFIX/$1/preprocessed";
	rm -r $PROCESS_DIR;
	mkdir -p $PROCESS_DIR;

	fairseq-preprocess --source-lang source --target-lang target \
	--trainpref input/$1/train --validpref input/$1/validation --testpref input/$1/test \
	--destdir $PROCESS_DIR
}

train(){

	PROCESS_DIR="$MODEL_DIR/$PREFIX/$1/preprocessed";
	MDL_DIR="$MODEL_DIR/$PREFIX/$1/model";
	rm -r $MDL_DIR;
	mkdir -p $MDL_DIR;

	CUDA_VISIBLE_DEVICES=1 \
	fairseq-train $PROCESS_DIR \
	--lr 0.1 --clip-norm 0.1 --dropout 0.1 --max-tokens 25000  --batch-size 96 \
    --arch fconv_jp_current --save-dir $MDL_DIR --fp16
}

evals() {

	MDL_DIR="$MODEL_DIR/$PREFIX/$1/model";
	PROCESS_DIR="$MODEL_DIR/$PREFIX/$1/preprocessed";

	fairseq-preprocess --source-lang source --target-lang target --testpref input/$2 --destdir tmp \
		--tgtdict $PROCESS_DIR/dict.target.txt --srcdict $PROCESS_DIR/dict.source.txt

	OUTPUT_DIR=tmp/$1
	mkdir -p $OUTPUT_DIR

	touch $OUTPUT_DIR/gen.out

	touch $OUTPUT_DIR/gen.out.org
	touch $OUTPUT_DIR/gen.out.ref
	touch $OUTPUT_DIR/gen.out.sys

	CUDA_VISIBLE_DEVICES=1 fairseq-generate --path $MDL_DIR/checkpoint_best.pt --batch-size 128 --results-path output/tmp \
	--fp16 tmp >  $OUTPUT_DIR/gen.out

	grep ^S $OUTPUT_DIR/gen.out > $OUTPUT_DIR/gen.out.org
	grep ^T $OUTPUT_DIR/gen.out > $OUTPUT_DIR/gen.out.ref
	grep ^H $OUTPUT_DIR/gen.out > $OUTPUT_DIR/gen.out.sys

	python main.py sort_sentences input_file=$OUTPUT_DIR/gen.out.sys output_file=comparison/$3/$PREFIX\_$1.out
}

echo $SCRIPTPATH

if [[ $FUNCTION == 'preprocess' ]]; then
	preprocess $2
elif [[ $FUNCTION == 'train' ]]; then
	train $2
elif [[ $FUNCTION == 'eval' ]]; then
	evals $2 $3 $4
else
echo "ERROR"
fi
