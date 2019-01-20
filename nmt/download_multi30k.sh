# Downloads Multi30K Data
# Based on https://github.com/salesforce/cove/tree/master/OpenNMT-py

mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz && tar -xf mmt_task1_test2016.tar.gz -C data/multi30k && rm mmt_task1_test2016.tar.gz

for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tools/tokenizer.perl -a -no-escape -l $l -q  < $f > $f.tok; done; done
# for l in en de; do for f in data/multi30k/*.$l; do perl tools/tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; done; done

for f in data/multi30k/*.tok; do perl tools/lowercase.perl < $f > $f.low; done  # To lowercase all data
python preprocess.py -train_src data/multi30k/train.en.tok.low -train_tgt data/multi30k/train.de.tok.low -valid_src data/multi30k/val.en.tok.low -valid_tgt data/multi30k/val.de.tok.low -save_data data/multi30k/multi30k.tok.low -lower
