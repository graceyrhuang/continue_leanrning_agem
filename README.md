# A-GEM with CGAN

## Table of Contents

- [Background](#background)
- [Environment](#environment)
- [HOWTO](#howto)

## Background
In lifelong learning, the learner is presented with a sequence of tasks, incrementally building a data-driven prior, which may be leveraged to speed up learning a new task. [Gradient Episodic Memory (GEM)](https://github.com/facebookresearch/GradientEpisodicMemory) provides a creative method that alleviates forgetting while allowing beneficial transfer of knowledge to previous tasks.[Averaged GEM (A-GEM)](https://github.com/facebookresearch/agem) is an improved version of GEM, which enjoys the same or even better performance as GEM, while being almost as computationally and memory efficient as EWC and other regularization based methods.
Both GEM and A-GEM have the same fixed-memory, which is used for previous tasks data storage. Our method proposed a new method to store the feature of previous tasks by using Conditional Conditional Generative Adversarial Nets (CGAN) to simulate past data. 

## Environment

* `Python 3.x`
* `Pytorch`

## HOWTO

A bash file is already provided to run script.

```shell
GPUID=-1
# -1 if only cpu is used
OUTDIR=outputs/permuted_MNIST_incremental_class
REPEAT=1
# A-GEM without GAN
python ./iBatchLearn_dual.py --gpuid ${GPUID}\
			     --repeat ${REPEAT} \
			     --incremental_class --optimizer SGD  \
			     --n_permutation 10 \
			     --force_out_dim 100 \
			     --schedule 2 \
			     --batch_size 128 \
			     --model_name MLP1000 \
			     --agent_type customization  
			     --agent_name AGEM_4000   \
			     --lr 0.1 \
			     --reg_coef 0.5
# A-GEM with CGAN
python ./iBatchLearn_dual.py --gpuid ${GPUID} \
			     --repeat ${REPEAT} \
                             --incremental_class \
                             --optimizer SGD \
                             --gan_add \
                             --n_permutation 10 \
                             --force_out_dim 100 \
                             --schedule 2 \
                             --batch_size 128 \
                             --model_name MLP1000 \
                             --agent_type customization \
                             --agent_name AGEM_4000 \
                             --lr 0.1 \
                             --reg_coef 0.5

```
