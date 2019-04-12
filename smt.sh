#  mkdir -p lm
#  touch lm/train.arpa.target
#  ~/git/mosesdecoder/bin/lmplz -o 3 <input/train.target > lm/train.arpa.target
#  ~/git/mosesdecoder/bin/build_binary \
#    lm/train.arpa.target \
#    lm/train.blm.target

# echo "これ は 日本語 の 文章 でしょ う か 。"                       \
#    | ~/git/mosesdecoder/bin/query lm/train.blm.target

# mkdir -p smt

# ~/git/mosesdecoder/scripts/training/train-model.perl -root-dir smt \
#  -corpus input/train                             \
#  -f source -e target -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
#  -lm 0:3:$HOME/Projects/japanese_gec/lm/train.blm.target:8                          \
#  -cores 4 \
#  -external-bin-dir ~/git/mosesdecoder/tools > training.out 2>&1 &


# ~/git/mosesdecoder/scripts/training/mert-moses.pl \
#   input/validation.source input/validation.target \
#   ~/git/mosesdecoder/bin/moses smt/model/moses.ini --mertdir ~/git/mosesdecoder/bin/ \
#   --decoder-flags="-threads 8" \
#   > mert.out 2>&1 &

 # ~/git/mosesdecoder/scripts/training/filter-model-given-input.pl             \
 #   filtered mert-work/moses.ini input/test.source \
 #   -Binarizer ~/git/mosesdecoder/bin/processPhraseTableMin

# mkdir -p comparison/test
# mkdir -p comparison/teacher
# mkdir -p comparison/student

 # ~/git/mosesdecoder/bin/moses            \
 #   -f filtered/moses.ini   \
 #   < input/test.source             \
 #   > comparison/test/smt.out       \
 #   2> gen.log.out 

 # ~/git/mosesdecoder/bin/moses            \
 #   -f filtered/moses.ini   \
 #   < input/teacher.source             \
 #   > comparison/teacher/smt.out       \
 #   2> gen.log.out 


 # ~/git/mosesdecoder/bin/moses            \
 #   -f filtered/moses.ini   \
 #   < input/student.source             \
 #   > comparison/student/smt.out       \
 #   2> gen.log.out 

  ~/git/mosesdecoder/bin/moses            \
   -f filtered/moses.ini   \
   < input/student_same.source             \
   > comparison/student_same/smt.out       \
   2> gen.log.out 

    ~/git/mosesdecoder/bin/moses            \
   -f filtered/moses.ini   \
   < input/teacher_same.source             \
   > comparison/teacher_same/smt.out       \
   2> gen.log.out 

# for i in {1..114}; do

# 	mkdir -p comparison/$i

# 	~/git/mosesdecoder/bin/moses            \
# 	   -f filtered/moses.ini  -threads all \
# 	   < input/$i/test.source             \
# 	   > comparison/$i/smt.out       \
# 	   2> gen.log.out 

# done

