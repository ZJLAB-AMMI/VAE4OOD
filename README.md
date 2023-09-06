# Error Reduction
Official implementation of [An Efficient Approach to Unsupervised Out-of-Distribution Detection with Variational Autoencoders](https://arxiv.org/abs/2309.02084).


## Citation
```
@misc{zeng2023efficient,
      title={An Efficient Approach to Unsupervised Out-of-Distribution Detection with Variational Autoencoders}, 
      author={Zezhen Zeng and Bin Liu},
      year={2023},
      eprint={2309.02084},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Training

Here we provide the VAE trained on CIFAR-10 with nearest interpolation as an example, for mask operation, check `utils.py` for the function. 
To train the VAE, use appropriate arguments and run this command. 

```train
python train_one.py 
```

| Argument | Purpose |
|:------:|:-------:|
| --dataroot | path to dataset |
| --batchSize | input batch size |
| --niter | number of training epochs |
| --nz | the dimension of the latent space |
| --lr | learning rate |
| --ratio | trains the backround model for Likelihood Ratios |
| --perturbed | mu hyperparameter for background model |

## Evaluation

To evaluate ER's OOD detection performance, run

```eval
python measure.py
```
| Argument | Purpose |
|:------:|:-------:|
| --dataroot | path to dataset |
| --batchSize | input batch size |
| --nz | the dimension of the latent space |
| --lam| factor for image complexity (lambda) |
| --repeat | repeat for each sample, only used for IWAE |
| --state_E | path to encoder checkpoint |
| --state_G | path to decoder checkpoint |


Above commands will save the numpy arrays containing the OOD scores for in-distribution and OOD samples in specific location, and to compute aucroc score, run
```eval
python aucroc.py
```

Code is modified from [Likelihood Regret]https://github.com/XavierXiao/Likelihood-Regret.


