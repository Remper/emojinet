# emojinet
The code of the experiments for the EVALITA2018 challenge

Try it (semeval files should be in `input/semeval_<test/train>.<labels/text>` with respect to data folder):
```
python src/train.py \
        --workdir data \
        --max-dict 100000 \
        --max-epoch 40 \
        --semeval
```
