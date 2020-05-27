# Pytorch Implementation of Pre-text Invariant Representation Learning
This repository contains the pyotrch implementation of Pretext invariant representation learning (PIRL)
algorithm on STL10 dataset. PIRL was originally introduced by Misra et al, publication of which can be found [here](https://arxiv.org/abs/1912.01991).

## What is PIRL and why is it useful
Pretext invariant representation learning (PIRL) is a self supervised learing algorithm that exploits contrastive
learning to learn visual representations such that original and transformed version of the same image have similar
representations, while being different from that of other images, thus achieving invariance to the transformation.

In their paper, authors have primarily focused on jigsaw puzzles transformation.

## Loss Function and slight modification
The CNN used for representation learning is trained using NCE (Noise Contrastive Estimation) technique,
NCE models the porbability of event that (I, I_t) (original and transformed image) originate from the same
data distribution, I.e.
![alt text](https://docs.google.com/drawings/d/e/2PACX-1vQIBzisD1g6le_VQlfj7oeJVr98inlrBsvTzssW35MO1nxilwXa2MhkUukLli1U1Orb50_kC_XY3XCL/pub?w=480&h=96 "probability function")
<br/>
Where s(., .) is cosine similarity between v_i and v_i_t, deep representations for original and transformed image respectively.
While, the final NCE loss is given as:
![alt text](https://docs.google.com/drawings/d/e/2PACX-1vRh2RjlYsPaSyGDORVN3zDl3sZ1r1g48jxW-fT8ajrGFx1rbHqyRnlepbZ63wr1K0oOCfjfndUhKA4S/pub?w=960&h=720 "L_nce")
where f(.) and g(.) are linear function heads.

## Slight Modification
Instead of using NCE loss, for this implementation, optimization process would directly aim to minimize
the negative log of probability described in the first equation above (with inputs as f(v_i) and g(v_i_t))

## Dataset Used
The implementation uses STL10 dataset, which can be downloaded from [here](http://ai.stanford.edu/~acoates/stl10/)
#### Dataset setup steps
```
1. Download raw data from above link to ./raw_stl10/
2. Run stl10_data_load.py. This will save three directories train, test and unlabelled in ./stl10_data/
```

## Training and evaluation steps
1. Run script pirl_stl_train_test.py for unsupervised (self supervised learning), example
```
python pirl_stl_train_test.py --model-type res18 --batch-size 128 --lr 0.1 --experiment-name exp
```
2. Run script train_stl_after_ssl.py for fine tuning model parameters obtained from self supervised learning, example
```
python train_stl_after_ssl.py --model-type res18 --batch-size 128 --lr 0.1  --patience-for-lr-decay 4 --full-fine-tune True --pirl-model-name <relative_model_path from above run>
```

## Results
After training the CNN model in PIRL manner, to evaluate how well learnt model weights transfer to classification
problem in limited dataset scenario, following experiments were performed.

Fine tuning strategy | Val Classification Accuracy
--- | ---
Only softmax layer is fine tuned |  50.50
Full model is fine tuned | 67.87

# References
1. PIRL paper: https://arxiv.org/abs/1912.01991
2. STL 10 dataset: http://ai.stanford.edu/~acoates/stl10/
3. Data loading code for STL 10: https://github.com/mttk/STL10
