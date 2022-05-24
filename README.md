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

Pretrained models are available at [google drive]().

## Train models

Logs will be saved in `logs/{dataset}_{model}_{arch}_b{method}` directory.

### Step 1. Train classifier model 

Train 
(We provide additional codes for our baseline models, which were taken from the author's code or reproduced directly. So the reproduced performance may be slightly different from the reported performance of each method.)
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












































========================================================================
Step1. Noisy Data Generation

* generate_noisy_data.py : generate synthetic noisy dataset
- If you run generate_noisy_data.py, you will get synthetic noisy labeled dataset.
- To save your time, we supply you with data.tar.gz at dropbox

========================================================================
Step2. Train a classifier
- To save your time, we supply you with classifier_model.tar.gz at dropbox

* train_classifier.py : train a classifier
 - Ex. Train CE model with symmetric 20% MNIST dataset,
: python3 train_classifier.py --dataset MNIST -- noise_type sym --noisy_ratio 0.2 --class_method no_stop --seed 0 --data_dir XXXXX

 - Note
	- For noise type, clean is no noise, sym is symmetric(SN), asym is asymmetric(ASN), idnx is instance-dependent(IDN) and idn is Similarity Related Instance-dependent (SRIDN) noise.
	- For class_method, refer to the code for the implemented model name (e.g. vanilla for Early Stopping)
	- you have to fill data_dir argument with the directory you saved the data(pickle files we supplied should be located in the directory).

========================================================================
Step3. Calibrate with NPC

* main_prior.py : KNN Prior
 - Ex. Train CE model with symmetric 20% MNIST dataset,
: python3 main_prior.py --dataset MNIST --noise_type sym --noisy_ratio 0.2 --class_method no_stop --seed 0 --data_dir XXXXXXX

* main_npc.py : NPC and other post-processing methods(RoG, KNN)
 - Ex. Train CE model with symmetric 20% MNIST dataset,
: python3 main_npc.py --dataset MNIST --noise_type sym --noisy_ratio 0.2 --class_method no_stop --post_method npc --knn_mode onehot --prior_norm 5 --data_dir XXXXX

========================================================================
Etc. Post-processing version of other possible methods
################## 없어도 괜찮을 거 같기도 하고...

* main_post.py : post-processors of star version
 - Ex. Train CE model with symmetric 20% MNIST dataset,
: python3 main_npc.py --dataset MNIST --noise_type sym --noisy_ratio 0.2 --class_method no_stop --post_processor causalnl --data_dir XXXXX


========================================================================
* 파일 제목 대문자는 classifier model, 소문자는 post-processor
