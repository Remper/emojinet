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
|BASE CNN                    |ft-it-our-100d |0.4435     |0.4413     |0.2354     |0.2560     |
|BASE CNN                    |provided       |0.4230     |0.4724     |0.1905     |0.2083     |
|BASE CNN                    |ft-it-300d     |0.4351     |0.3489     |0.2464     |0.2673     |
|BASE LSTM                   |ft-it-our-100d |0.4443     |0.3666     |0.2586     |0.2809     |
|BASE LSTM                   |ft-it-300d     |0.4053     |0.3167     |0.2534     |0.2707     |
|BASE LSTM                   |provided       |0.4415     |0.3836     |0.2408     |0.2622     |
|Most frequent user history  |—              |0.4396     |0.4076     |0.2774     |0.3133     |
|BASE LSTM User              |ft-it-our-100d |0.4874     |0.4343     |0.3218     |0.3565     |
|BASE LSTM User (userdata)   |ft-it-our-100d |**0.5153** |0.4642     |**0.3477** |**0.3840** |

### Experiment log

| Model                      |Split    |Embeddings     | Accuracy  | Precision | Recall    | F1        | Remarks           |
|:---------------------------|:------- |:------------- |:--------- |:--------- |:--------- |:--------- |:----------------- |
|BASE LSTM User              |42 (def) |ft-it-our-100d |0.4875     |0.4333     |0.3242     |0.3575     | dict size: 100000 |
|BASE LSTM User              |42 (def) |ft-it-our-100d |0.4885     |0.4362     |0.3220     |0.3571     | dict size: 100000 |
|BASE LSTM User              |42 (def) |ft-it-our-100d |0.4863     |0.4335     |0.3193     |0.3548     | dict size: 100000 |
|BASE LSTM                   |42 (def) |ft-it-our-100d |0.4444     |0.3634     |0.2626     |0.2852     |                   |
|BASE LSTM                   |42 (def) |ft-it-our-100d |0.4428     |0.3586     |0.2638     |0.2831     |                   |
|BASE LSTM                   |42 (def) |ft-it-our-100d |0.4458     |0.3779     |0.2494     |0.2743     |                   |
|BASE LSTM                   |42 (def) |provided       |0.4381     |0.3662     |0.2481     |0.2701     |                   |
|BASE LSTM                   |42 (def) |provided       |0.4456     |0.3999     |0.2440     |0.2673     |                   |
|BASE LSTM                   |42 (def) |provided       |0.4409     |0.3847     |0.2303     |0.2492     |                   |
|BASE LSTM                   |42 (def) |ft-it-300d     |0.3974     |0.3102     |0.2616     |0.2749     |                   |
|BASE LSTM                   |42 (def) |ft-it-300d     |0.4157     |0.3321     |0.2502     |0.2721     |                   |
|BASE LSTM                   |42 (def) |ft-it-300d     |0.4027     |0.3077     |0.2483     |0.2650     |                   |