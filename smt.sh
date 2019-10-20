SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
MODEL_DIR='model'

PREFIX='smt'

FUNCTION=$1
echo $FUNCTION

echo "$MODEL_DIR/lm"


make_lm(){

	python mv_input.py $1;

	LM_DIR="$MODEL_DIR/$PREFIX/$1/lm";
	rm -r $LM_DIR;
	mkdir -p $LM_DIR;

 	touch $LM_DIR/train.arpa.target
 	~/git/mosesdecoder/bin/lmplz -o 3 <input/$1/train.target > $LM_DIR/train.arpa.target;
 	~/git/mosesdecoder/bin/build_binary \
   	$LM_DIR/train.arpa.target \
   	$LM_DIR/train.blm.target;

	echo "これ は 日本語 の 文章 でしょ う か 。"                       \
	   | ~/git/mosesdecoder/bin/query $LM_DIR/train.blm.target;
}

train(){

	LM_DIR="$MODEL_DIR/$PREFIX/$1/lm";
	MDL_DIR="$MODEL_DIR/$PREFIX/$1/model";
	rm -r $MDL_DIR;
	mkdir -p $MDL_DIR;

	~/git/mosesdecoder/scripts/training/clean-corpus-n.perl input/$1/train source target input/$1/filtered_train 1 100;
	~/git/mosesdecoder/scripts/training/train-model.perl -root-dir $MDL_DIR \
	 -corpus input/$1/train                             \
	 -f source -e target -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
	 -lm 0:3:$SCRIPTPATH/$LM_DIR/train.blm.target:8                          \
	 -cores 10 \
	 -external-bin-dir ~/git/mosesdecoder/tools;
}

tune(){

	MDL_DIR="$MODEL_DIR/$PREFIX/$1/model"
	TUNE_DIR="$MODEL_DIR/$PREFIX/$1/tune"
	rm -r $TUNE_DIR
	mkdir -p $TUNE_DIR
	~/git/mosesdecoder/scripts/training/mert-moses.pl \
	  input/$1/validation.source input/$1/validation.target \
	  ~/git/mosesdecoder/bin/moses $MDL_DIR/model/moses.ini  \
	  --mertdir ~/git/mosesdecoder/bin/ \
	  --decoder-flags="-threads 8" --batch-mira --return-best-dev;
}

_eval() {

	MDL_DIR="$MODEL_DIR/$PREFIX/$1/model/model"

	mkdir -p comparison/$2
		~/git/mosesdecoder/bin/moses \
		-f $MDL_DIR/moses.ini  -threads all \
	   	< input/$2/fixed.source \
	   	> comparison/$2/$PREFIX\_$1.out
}

_eval_rule() {


for i in {1..125}; do

	mkdir -p comparison/$2/$i

	cp -r input/$2/$i/ comparison/$2/


	echo "Processing RULE: $i"

	~/git/mosesdecoder/bin/moses \
		-f $MDL_DIR/moses.ini  -threads all \
	   	< comparison/$2/$i/test.source \
	   	> comparison/$2/$i/$PREFIX\_$1.out;
done

}


echo $SCRIPTPATH

if [[ $FUNCTION == 'make_lm' ]];
then
	make_lm $2
elif [[ $FUNCTION == 'train' ]]; then
	train $2
elif [[ $FUNCTION == 'tune' ]]; then
	tune $2
elif [[ $FUNCTION == 'eval' ]]; then
	_eval $2 $3
elif [[ $FUNCTION == 'eval_rule' ]]; then
	_eval_rule $2 $3
else
	echo "ERROR"
fi
