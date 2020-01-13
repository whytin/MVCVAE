# MVCVAE

This is a Keras implementation of “Shared Generative Latent Representation Learning for Multi-view Clustering”

# Requirements

Python 3
Keras 1.1.0
Theano 1.0.3
scikit-learn 0.17.1

# Preliminary

1. Download dataset and saved models from Baidu cloud: 

URL：https://pan.baidu.com/s/1ewL2Krzog3g-f-fWsGAPgg 

PASSWORD：zhsw 

2. Move downloaded file to MVCVAE/dataset and MVCVAE/MVCVAE_pretrain respectively.

3. Replace tools/initializations.py to /where/keras/installed/directory/initializations.py

Replace tools/training.py to /where/keras/installed/directory/engine/training.py

# Train

Train the specific dataset by adding the argument --dataset. Optional dataset ('UCI2', 'NUS5', 'caltech7', 'UCI6', 'Cal2', 'ORL3', 'ORL2')
```
python MVCVAE_train.py --dataset 'NUS5'
```


























































































































