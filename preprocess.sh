python mv_file.py

DIR_DATA=input

fairseq-preprocess --source-lang source --target-lang target \
	--trainpref $DIR_DATA/train --validpref $DIR_DATA/validation --testpref $DIR_DATA/test \
	--destdir $DIR_DATA/processed
