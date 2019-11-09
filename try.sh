
FUNCTION=$1
INPUT_FOLDER=$2
DATAFILE=$3
OUTPUT_FOLDER=$4
MODEL_TYPE=$5
MODEL=$6

mkdir -p comparison/$OUTPUT_FOLDER/tmp

if [[ $FUNCTION == 'process' ]];
then
	python main.py pre_process_delimited_txt \
	   input_file=raw_data/$INPUT_FOLDER/$DATAFILE\_cleaned.txt \
	   output_source=input/$INPUT_FOLDER/$DATAFILE.source \
	   output_target=input/$INPUT_FOLDER/$DATAFILE.target \
	   output_start=input/$INPUT_FOLDER/$DATAFILE.start \
	   raise_on_error=True
	   # output_start=input/$INPUT_FOLDER/$DATAFILE.start \
	   # cleaned_file=raw_data/$INPUT_FOLDER/$DATAFILE\_cleaned.txt

	cp input/$INPUT_FOLDER/$DATAFILE.source comparison/$INPUT_FOLDER/$DATAFILE.source
	cp input/$INPUT_FOLDER/$DATAFILE.target comparison/$INPUT_FOLDER/$DATAFILE.target

elif [[ $FUNCTION == 'classify' ]]; then

	python main.py match_parallel_text_rules \
		rule_file=new \
		input_source=input/$INPUT_FOLDER/$DATAFILE.source \
		input_target=input/$INPUT_FOLDER/$DATAFILE.target \
		input_start=input/$INPUT_FOLDER/$DATAFILE.start \
		rule_index=-1 \
		language_dir=database/full \
		unique_dir=database/full \
		output_dir=comparison/$OUTPUT_FOLDER \
		print_unmatched=False

elif [[ $FUNCTION == 'run' ]]; then

	if [[ $MODEL_TYPE == 'smt' ]];
	then
		./smt.sh eval $MODEL $INPUT_FOLDER/$DATAFILE $OUTPUT_FOLDER/tmp;
	elif [[ $MODEL_TYPE == 'fairseq' ]]; then
		./fairseq.sh eval $MODEL $INPUT_FOLDER/$DATAFILE $OUTPUT_FOLDER/tmp;
	else
		echo "ERROR"
	fi

	python main.py split_eval_data corpus_name=$OUTPUT_FOLDER;
	# python main.py eval_binary_corpus corpus_name=$OUTPUT_FOLDER;
else

	echo "ERROR"
fi

