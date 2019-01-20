# Downloads the IWSLT 2016 Data
wget https://wit3.fbk.eu/archive/2016-01//texts/en/de/en-de.tgz -P data/
tar -xzf data/en-de.tgz -C data/
rm data/en-de.tgz
python clean.py
for l in en de; do for f in data/en-de/*.$l; do perl tools/tokenizer.perl -a -no-escape -l $l -q  < $f > $f.tok; done; done
for f in data/en-de/*.tok; do perl tools/lowercase.perl < $f > $f.low; done  # To lowercase all data
python preprocess.py -train_src data/en-de/train.en-de.en.tok.low -train_tgt data/en-de/train.en-de.de.tok.low -valid_src data/en-de/IWSLT16.TED.tst2013.en-de.en.tok.low -valid_tgt data/en-de/IWSLT16.TED.tst2013.en-de.de.tok.low -save_data data/en-de/en-de.tok.low -lower


# IWSLT De-En
# Based on https://github.com/salesforce/cove/tree/master/OpenNMT-py

# mkdir -p data/iwslt16
# wget https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz && tar -xf de-en.tgz -C data
# python iwslt_xml2txt.py data/de-en
# python iwslt_xml2txt.py data/de-en -a
# python preprocess.py -train_src data/de-en/train.de-en.en.tok -train_tgt data/de-en/train.de-en.de.tok -valid_src data/de-en/IWSLT16.TED.tst2013.de-en.en.tok -valid_tgt data/de-en/IWSLT16.TED.tst2013.de-en.de.tok -save_data data/iwslt16.tok.low -lower -src_vocab_size 22822 -tgt_vocab_size 32009

# python train.py -data data/iwslt16.tok.low.train.pt  -save_model somesave -gpus 0 -brnn -rnn_size 600
# python translate.py -gpu 0 -model model_name -src data/de-en/IWSLT16.TED.tst2014.de-en.en.tok -tgt data/de-en/IWSLT16.TED.tst2014.de-en.de.tok -replace_unk -output iwslt.ted.tst2014.de-en.tok.low.pred
# perl multi-bleu.perl data/de-en/IWSLT16.TED.tst2014.de-en.de.tok < iwslt.ted.tst2014.de-en.tok.low.pred
