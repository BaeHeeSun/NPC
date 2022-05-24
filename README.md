# NPC (Noisy Prediction Calibration via Generative Model)

Official PyTorch implementation of
[**"From Noisy Prediction to True Label: Noisy Prediction Calibration via Generative Model"**](https://arxiv.org/abs/2205.00690) (ICML 2022) by
[HeeSun Bae*](https://scholar.google.co.kr/citations?hl=ko&view_op=list_works&gmla=AJsN-F47spNJkOB5PqPd5qYdvduZMN7Jp9ppZsN5FPpfX71F4fdliD29eOlFOktmElm9o59IBMc3xwUuM0oDMmw9yH4lx66lCJ3tjBsiVNu6RpjVTO8e0t-Ul2d1AJpbhoUr3gyvs4Dp&user=D9U_ohsAAAAJ),
[Seungjae Shin*](https://sites.google.com/view/seungjae-shin),
[Byeonghu Na](https://wp03052.github.io/),
JoonHo Jang,
[Kyungwoo Song](https://mlai.uos.ac.kr/),
and [Il-Chul Moon](https://aailab.kaist.ac.kr/xe2/members_professor/6749).

## Setup

Install required libraries.
```
pip install -r requirements.txt
```
Download datasets in `/data`.

Pretrained models are available at [dropbox]().

## Train models

Logs will be saved in `logs/{dataset}_{model}_{arch}_b{method}` directory.

### Step 0. Noisy Data Generation

Generate synthetic noisy dataset.
```
python generate_noisy_data.py
```
We also provide pre-processed dataset at [dropbox]().

### Step 1. Train classifier model 

Train the classifier model. 
```
python train_classifier.py --dataset MNIST -- noise_type sym --noisy_ratio 0.2 --class_method no_stop --seed 0 --data_dir {your_data_directory}
```
It will train the base classifier with CE (Cross Entropy) loss on the MNIST dataset with `sym` (symmetric 20%) noise. 

We also provide other noise types:
* `clean` : no noise
* `sym` : symmetric
* `asym` : asymmetric
* `idnx` : instance-dependent
* `idn` : Similarity related instance-dependent

Please refer the code for the notation of each pre-training method. (e.g. `vanilla` for early-stopping)
To save your time, We also provide the checkpoints of pre-trained classifiers at [dropbox]().

### Step 2. Compute prior from classifier model
Pre-compute prior information from classifier model
```
python xxxx.py
```

### Step 3. Train NPC to calibrate the prediction of pre-trained classifier
Train contextual debiased model with object-aware random crop.
```
python xxxx.py
```













































