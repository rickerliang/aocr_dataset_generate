#!/bin/bash

screen -dmS train bash -c 'source ~/.bash_profile; cd $0; python train.py --batch_size=48 --dataset_name=han_dataset_1 --train_log_dir=./aocr; exec bash' $PWD
screen -dmS eval_train bash -c 'source ~/.bash_profile; cd $0; python eval.py  --dataset_name=han_dataset_1 --train_log_dir=./aocr --eval_log_dir=./aocr_eval_train; exec bash' $PWD
screen -dmS eval_test bash -c 'source ~/.bash_profile; cd $0; python eval.py  --dataset_name=han_dataset_1 --train_log_dir=./aocr --eval_log_dir=./aocr_eval_test --split_name=test; exec bash' $PWD
screen -dmS tensorboard_train bash -c 'source ~/.bash_profile; cd $0; tensorboard --logdir=./aocr --port=6006' $PWD
screen -dmS tensorboard_eval_train bash -c 'source ~/.bash_profile; cd $0; tensorboard --logdir=./aocr_eval_train --port=6007' $PWD
screen -dmS tensorboard_eval_test bash -c 'source ~/.bash_profile; cd $0; tensorboard --logdir=./aocr_eval_test --port=6008' $PWD
 