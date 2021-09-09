python data/process.py \
  --data_name='yahoo' \
  --sup_file='data/new_sup' \
  --unsup_file='data/new_unsup' \
  --test_file='data/new_test' \
  --sup_label_num=500 \
  --unsup_label_num=70000 \
  --test_label_num=5000 \
  --seq_len=256 \
  $@