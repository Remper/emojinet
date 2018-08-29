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

| Model                      |Embeddings     | Accuracy  | Precision | Recall    | F1        |
|:---------------------------|:------------- |:--------- |:----------|:----------|:----------|
|Ensemble CNN subword        |ft-it-our-100d |0.4375     |0.3357     |0.2603     |0.2737     |
|VDCNN 9                     |—              |0.3431     |0.2343     |0.1750     |0.1824     |
|VDCNN 9 shortcut            |—              |0.3673     |0.2943     |0.1772     |0.1926     |
|VDCNN 9 shortcut dropout    |—              |0.3656     |0.2997     |0.1428     |0.1399     |
|BASECNN                     |ft-it-our-100d |0.4435     |0.4413     |0.2354     |0.2560     |
|BASECNN                     |provided       |0.4230     |0.4724     |0.1905     |0.2083     |
|BASECNN                     |ft-it-300d     |0.4351     |0.3489     |0.2464     |0.2673     |
|BASE LSTM                   |ft-it-our-100d |0.4428     |0.3586     |0.2638     |0.2831     |
|BASE LSTM                   |provided       |0.4381     |0.3662     |0.2481     |0.2701     |
|BASE LSTM                   |ft-it-300d     |0.4027     |0.3077     |0.2483     |0.2650     |
|Most frequent user history  |—              |0.4396     |0.4076     |0.2774     |0.3133     |
|BASE LSTM User              |ft-it-our-100d |**0.4863** |**0.4335** |**0.3193** |**0.3548** |

