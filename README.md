# Topological Neural Networks go Persistent, Equivariant and Continuous

 [Yogesh Verma](https://yoverma.github.io/yoerma.github.io/) | [Amauri H. Souza](https://www.amauriholanda.org)  |  [Vikas Garg](https://www.mit.edu/~vgarg/)

The repository is developed on the intersection of [RePHINE](https://github.com/Aalto-QuML/RePHINE), [TOGL](https://github.com/BorgwardtLab/TOGL), [EMPSN](https://arxiv.org/abs/2305.07100) and [AbODE](https://github.com/yogeshverma1998/AbODE). Please refer to their repos for specific requirements.


## Prerequisites

- Please install all the package requirements mentioned in RePHINE, TOGL, and AbODE.
- torchdiffeq: https://github.com/rtqichen/torchdiffeq.


## Training

### Graph Classification


#### Comparison with RePHINE

```
cd RePHINE/
python -u main_2d.py  --dataset #######  --gnn {gin/gcn} --diagram_type {standard/rephine}  --nsteps 20 
```


#### Comparison with TOGL

```
cd RePHINE/
python -u main_togl.py --dataset {ENZYMES/DD/Proteins} --gnn {gin/gcn}
```


### QM9 Property Prediction


```
cd empsn/
python -u main_qm9.py --target_name {mu,alpha,gap,r2,zpve,Cv,homo,lumo} --epochs 1000 --dis 4.0 --dim 2 --num_hidden 77 --seed 42 --model_name {empsn_rephine_cont/empsn_rephine}
```


### CDR-H3 Antibody Design

Download the train/test/val files from here . Kindly add the paths to these files in ```train_topnets.py``` file. 

```
cd Antibody/
python -u train_topnets.py --cdr 3
```
