TASK="ed"
COMPLEXITY=16

MODEL="GPT"
LAYER=2
LOOP=8
COT_LENGTH=16

python -m experiments.train \
 --task ${TASK}\
 --input_size ${COMPLEXITY}\
 --model ${MODEL}\
 --n_layer ${LAYER}\
 --n_loop ${LOOP}\
 --epoch 200 \
 --batch_size 256 \
 --cot_length ${COT_LENGTH} \
 --chain \
 --val_interval 100 \
 