python train.py --task=ace05 \
                --data_dir=/path/to/pure_style_data \
                --output_dir=./model \
                --train_batch_size=16 \
                --learning_rate=1e-5  --task_learning_rate=5e-4 \
                --context_window=50 \
                --model=path/to/pretrained_model \
                --do_train --do_eval --eval_test