# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# https://stackoverflow.com/questions/2107945/how-to-loop-over-directories-in-linux
data_dir=$1
output_dir=$2

echo "Using Data From ${data_dir}, Saving to ${output_dir}"

python -m torch.distributed.launch \
    --nproc_per_node 4 run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mnli \
    --do_train \
    --do_eval \
    --data_dir ${data_dir} \
    --max_seq_length 128 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 10.0 \
    --output_dir ${output_dir} \
    --weight_decay 0.005 \
    --save_steps 5000 \
    --logging_steps 100 \
    --save_total_limit 1