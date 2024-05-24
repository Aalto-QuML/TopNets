# Topological Neural Networks go Persistent, Equivariant and Continuous

 [Yogesh verma](https://yoverma.github.io/yoerma.github.io/) | [Amauri H. Souza](https://www.amauriholanda.org)  |  [Vikas Garg](https://www.mit.edu/~vgarg/)

The repository is developed on the intersection of [RePHINE](https://github.com/Aalto-QuML/RePHINE), [TOGL](https://github.com/BorgwardtLab/TOGL) and [AbODE](https://github.com/yogeshverma1998/AbODE). Please refer to their repos for specific requirements.


## Prerequisites

- Please install all the package requirements as mentioned in RePHINE, TOGL, and AbODE.
- torchdiffeq : https://github.com/rtqichen/torchdiffeq.
- pytorch >= 1.12.0
- pytorch_geometric >= 1.12.0


## Training

### Graph Classification


#### Comparison with RePHINE

```
cd RePHINE/
python -u main_2d.py  --dataset #######  --gnn gin --diagram_type {standard/rephine}  --nsteps 20 
```


#### Comparison with TOGL

```
cd RePHINE/
python -u main_togl.py --dataset {ENZYMES/DD/Proteins}
```


### CDR-H3 Antibody Design

```
cd Antibody/
python -u train_topnets.py --cdr 3
```
