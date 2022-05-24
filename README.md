# NPC (Noisy Prediction Calibration via Generative Model)

Official PyTorch implementation of
[**"From Noisy Prediction to True Label: Noisy Prediction Calibration via Generative Model"**](https://arxiv.org/abs/2205.00690) (ICML 2022) by
HeeSun Bae*,
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

Pretrained models are available at [google drive]().

## Train models

Logs will be saved in `logs/{dataset}_{model}_{arch}_b{method}` directory.

### Step 1. Train classifier model 

Train 
```
python xxxx.py
```

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

