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

### Data analysis

```
python3 data_analysis/data_analysis.py 
        --workdir evalita_data
```

This script will output a file containing total number, unique number and distributions of:
* tokens
* hashtags
* mentions
* URLS
* labels

### VDCNN

```
python3 train_vdcnn.py 
        --workdir evalita_data
        --max-seq-length 512
        --pool-type k_max
        --depth 29
        --shortcut
        --bias

```
