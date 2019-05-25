# Zero Shot Knowledge Transfer

This is accompanying code for the paper "Zero-shot Knowledge Transfer via Adversarial Belief Matching" [see arxiv](https://arxiv.org/abs/1905.09768)

## What is this work about?

![picture](images/butterfly.gif)

**TLDR:** Our task is to compress a large neural network (teacher) into a smaller one (student), but we assume that the data used to train the teacher is not available anymore. We thus generate data adversarially (yellow markers above) and use that for distillation instead.

## Environment
- Python 3.6
- pytorch 1.0.0 (both cpu and gpu version tested)
- tensorboard 1.7.0 (for logging, + needs tensorflow)

## Run zero shot knowledge transfer
1. Pretrain a teacher for the dataset/architecture you want, or download some of mine [here](https://drive.google.com/drive/folders/1lLgAndtJGUOUWvFGC8f1BFA5RIgyEfct?usp=sharing)
2. Make sure you have the same folder structure as in the link above, i.e. Pretrained/{dataset}/{architecture}/last.pth.tar
3. Edit the paths in scripts/ZeroShot/main0.sh and run it

## Make transition curves
1. Pretrain a zero-shot student or a student with KD+AT
2. Edit the paths in scripts/TransitionCurves/transition_curves0.sh and run it


## Cite
This work is under review for NeurIPS2019. In the mean time, please cite:
```
@article{Micaelli2019ZeroShotKT,
  author    = {Paul Micaelli and
               Amos Storkey},
  title     = {Zero-shot Knowledge Transfer via Adversarial Belief Matching},
  url       = {https://arxiv.org/abs/1905.09768}
}

```
