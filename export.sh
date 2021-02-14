BERT_DIR=`pwd`
export BERT_BASE_DIR=${BERT_DIR}/chinese_L-12_H-768_A-12
export DATA_DIR=${BERT_DIR}/data

python run_classifier.py \
  --task_name=sentiment \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=output \
  --do_export=true \
  --export_dir=exported
