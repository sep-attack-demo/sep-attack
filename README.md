# SEP-Attack: A Simple and Effective Paradigm for Transfer-based Textual Adversarial Attack

## Requirements
```
conda create -n sepattack python==3.9.1
```
then
```
pip install -r requirements.txt
```


## Download Dependencies
- Download the pretrained target models from [here](https://github.com/SEP-Attack/SEP-Attack-demo)

- Download the counter-fitted-vectors from [here](https://drive.google.com/file/d/1bayGomljWb6HeYDMTDKXrh0HackKtSlx/view) and place it in the main directory.

- Download top 50 synonym file from [here](https://drive.google.com/file/d/1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx/view) and place it in the main directory.

- Download the glove 300 dimensional vectors from [here](https://nlp.stanford.edu/projects/glove/) unzip it. **And change the 21-th line of SEP_utils.py**

## How to Run:
Use the following command to get the rsults.


```python3
python3 SEP_Attack.py \
        --dataset_path data/ag \ # or data/mr, data/yelp, data/imdb
        --word_embeddings_path glove.6B.200d.txt \
        --target_model cnn_tradi \ # or bert lstm_tradi
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset ag \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --k 1\
        --save_path ensemble/ag/cnn/ \
        --bstart 0 \
        --bend 500 
```