BERT_DIR=`pwd`
export BERT_BASE_DIR=${BERT_DIR}/chinese_L-12_H-768_A-12

python ./extract_features.py \
  --input_file=data/input.txt \
  --output_file=data/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --layers=-2 \
  --batch_size=8
