# REFER: An End-to-end Rationale Extraction Framework for Explanation Regularization

This repository provides the implementation for [this](https://arxiv.org/abs/2310.14418) paper.

### Installing Dependencies
```
git clone https://github.com/qasemii/REFER.git
pip install torch==1.7.1 torchvision torchaudio torchtext -f https://download.pytorch.org/whl/cu101/torch_stable.html    
pip install -r REFER/requirements.txt
pip install tensorflow --upgrade
pip install neptune-client==0.14.2
pip install protobuf==3.20.*
```

### Download the datasets
```
mkdir -p data/eraser/cose/
wget https://www.eraserbenchmark.com/zipped/cose.tar.gz
tar -xvzf cose.tar.gz -C data/eraser/cose/
mv -v data/eraser/cose/data/cose/* data/eraser/cose/
rm -rf data/eraser/cose/data
cd REFER/
```

### Build the datasets
```
python scripts/build_dataset.py --data_dir ../data --dataset cose --arch google/bigbird-roberta-base --split train
python scripts/build_dataset.py --data_dir ../data --dataset cose --arch google/bigbird-roberta-base --split train
python scripts/build_dataset.py --data_dir ../data --dataset cose --arch google/bigbird-roberta-base --split dev
python scripts/build_ood_dataset.py --data_dir ../data --dataset mnli_contrast_contrast --arch google/bigbird-roberta-base --split test
```

### Steps Before Running
1. Create a [Neptune](https://neptune.ai/) account and open a project.
2. Change `neptune-local-username` and `neptune-api-token` in [src/utils/logging.py](src/utils/logging.py) with your [Neptune](https://neptune.ai/) username and API tokens.
3. Go to [configs/logger/neptune.yaml](configs/logger/neptune.yaml) and change `neptune_project_name` with the project name you opened in [Neptune](https://neptune.ai/).


### TRAIN
Change `model.comp_wt`, `model.suff_wt`, and `model.plaus_wt` with your desired weights. Setting `model.e2e=True` includes AIMLE and makes the model end-to-end. 

```
HYDRA_FULL_ERROR=1   python   main.py -m   logger.offline=False  save_checkpoint=True   data=cose   model=expl_reg   model.explainer_type=self_lm   model.expl_head_type=linear   model.task_wt=1.0   model.comp_wt=1.0   model.suff_wt=1.0   model.plaus_wt=1.0   model.optimizer.lr=2e-5   setup.train_batch_size=4   setup.accumulate_grad_batches=8   setup.eff_train_batch_size=32   setup.eval_batch_size=32   setup.num_workers=3   seed=0   model.train_topk=[50]   model.eval_topk=[50]   +model.e2e=True  model.expl_reg_freq=0.1,1
```

### EVALUATE
```
HYDRA_FULL_ERROR=1   python   main.py -m   logger.offline=True   data=cose   model=expl_reg   model.explainer_type=lm   model.expl_head_type=linear   training=evaluate   training.eval_splits=train,dev,test   training.ckpt_path='../save/REF-1/checkpoints/epoch\=2-step\=51497.ckpt'  setup.eval_batch_size=32   setup.num_workers=3   seed=0   model.eval_topk=[50]   +model.e2e=True
```


