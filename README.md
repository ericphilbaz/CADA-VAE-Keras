# CADA-VAE-Keras

Keras implementation of "Generalized Zero-and Few-Shot Learning via Aligned Variational Autoencoders" (CVPR 2019)

- [Paper](https://arxiv.org/pdf/1812.01784.pdf)
- [Original PyTorch implementation](https://github.com/edgarschnfld/CADA-VAE-PyTorch)


## Differences to PyTorch implementation

- Losses are averaged (instead of summed) over each minibatch. This allows a more flexible batch size. For this, an additional weighting factor for the reconstruction loss is introduced.
- Generalized Few-Shot and non-generalized Zero-Shot are not implemented.

## Requirements

```
python==3.6.5
numpy==1.14.5
keras==2.2.4
tensorflow==1.12.0
scipy==1.1.0
```


## How to run the code
1. [Download data](https://www.dropbox.com/sh/btoc495ytfbnbat/AAAaurkoKnnk0uV-swgF-gdSa?dl=0) and specify data_dir in main.py
2. Optionally change params in main.py
3. Run main.py

## Experimental Protocol

- Test-accuracies are averaged over 8 runs (10 runs in paper)
- Hyperparameter search with random search and TPE (hyperopt library)

## Optimizing hyperparameter for CUB

Model | S | U | H
--- | --- | --- | ---
Paper                   | 53.5 | 51.6 | 52.4
pytorch-code          | 55.2 | 48.6 | 51.7
keras-code | 55.6 | 47.2 | 51.1

- exemplary fluctuation of test accuracies for pytorch_code: 51.2 to 52.5
- keras_code val-h-acc: 54.5

## Optimizing hyperparameter for all datasets

**Paper:**

Model   | S | U | H
---     | --- | --- | ---
CUB     | 53.5 | 51.6 | 52.4
SUN     | 35.7 | 47.2 | 40.6
AWA1     | 72.8 | 57.3 | 64.1
AWA2     | 75.0 | 55.8 | 63.9


**Optimizing for mean-val-h-accuracy over all datasets:**

Model   | S | U | H
---     | --- | --- | ---
CUB     | 55.0 | 47.6 | 51.0 +/- 0.5
SUN     | 35.1 | 42.7 | 38.5 +/- 0.4
AWA1     | 83.6 | 40.4 | 54.5 +/- 1.9
AWA2     | 86.6 | 41.2 | 55.8 +/- 2.0

- Mean val-h: 55.0

## Future Work

- Drop weighting factor for reconstruction loss, optimize learning rate instead
- Improve hyperparameter search over all datasets
- Implement Generalized Few-Shot
- Implement non-generalized Zero-Shot

Currently, the work on this repo is discontinued.