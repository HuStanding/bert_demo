BERT_DIR=`pwd`
export BERT_BASE_DIR=${BERT_DIR}/chinese_L-12_H-768_A-12
export OUTPUT_DIR=./output_squad
export SQUAD_DIR=${BERT_DIR}/squad_data

python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=${OUTPUT_DIR}

python ${SQUAD_DIR}/evaluate-v1.1.py squad_data/dev-v1.1.json ${OUTPUT_DIR}/predictions.json
