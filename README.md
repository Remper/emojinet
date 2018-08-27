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
        --evalita
        --workdir evalita_data
        --max-seq-length 1024
        --pool-type k_max
        --depth 29
        --shortcut
        --bias

```

### Results

| Model          | Accuracy | Precision | Recall | F1    |
|:---------------|:---------|:----------|:-------|:------|
|VDCNN 9         |0.3431    |0.2343     |0.1750  |0.1824 |
|VDCNN 9 shortcut|0.3673    |0.2943     |0.1772  |0.1926 |
|BASECNN fasttext|0.4435    |0.4413     |0.2354  |0.2560 |
|BASECNN provided|0.4230    |0.4724     |0.1905  |0.2083 |
|BASECNN 300d    |0.0000    |0.0000     |0.0000  |0.0000 |
