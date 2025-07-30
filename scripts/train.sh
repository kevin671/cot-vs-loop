TASK="path" # "arithmetic"
COMPLEXITY=32

MODEL="Looped"
LOOP=8
batch_size=1024

python -m experiments.train \
 --task ${TASK}\
 --input_size ${COMPLEXITY}\
 --model ${MODEL}\
 --n_loop ${LOOP}\
 --epoch 1000 \
 --val_interval 50 \
 --batch_size ${batch_size} \
 --curriculum geometric \