# Zero Shot Knowledge Transfer

This is accompanying code for the NeurIPS 2019 spotlight paper "Zero-shot Knowledge Transfer via Adversarial Belief Matching" [see arxiv](https://arxiv.org/abs/1905.09768)

## What is this work about? (TLDR)

![picture](images/butterfly.gif)

 Our task is to compress a large neural network (teacher) into a smaller one (student), but we assume that the data used to train the teacher is not available anymore. We thus generate pseudo points adversarially (yellow markers above) and use those to match the student (right) to the teacher (left).

## Environment
- Python 3.6
- pytorch 1.0.0 (both cpu and gpu version tested)
- tensorboard 1.7.0 (for logging, + needs tensorflow)
- scipy <= 1.2 (otherwise scipy.misc.toimage complains in logger)

## Run zero shot knowledge transfer
1. Pretrain a teacher for the dataset/architecture you want (or download some of mine [here](https://drive.google.com/drive/folders/1lLgAndtJGUOUWvFGC8f1BFA5RIgyEfct?usp=sharing))
2. Make sure you have the same folder structure as in the link above, i.e. Pretrained/{dataset}/{architecture}/last.pth.tar
3. Edit the paths in e.g. scripts/ZeroShot/main0.sh and run it

## Make transition curves
1. Pretrain a zero-shot student or a student distilled with KD+AT (or download some of mine [here](https://drive.google.com/drive/folders/1lLgAndtJGUOUWvFGC8f1BFA5RIgyEfct?usp=sharing))
2. Edit the paths in e.g. scripts/TransitionCurves/transition_curves0.sh and run it
3. This saves .pickle file with all the transition curves
4. You can import that file in notebooks/transition_curves_and_error.ipynb to plot the transition curves and calculate MTE scores

## Cite
If you use this work please consider citing:
```
@incollection{Micaelli2019ZeroShotKT,
title = {Zero-shot Knowledge Transfer via Adversarial Belief Matching},
author = {Micaelli, Paul and Storkey, Amos J},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {9551--9561},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/9151-zero-shot-knowledge-transfer-via-adversarial-belief-matching.pdf}
}
```

## Baselines

A few people have asked for the KD+AT few shot baselines so I've made them available in [this repo](https://github.com/polo5/FewShotKnowledgeTransfer)

## Notes
- Version 1 and 2 on arxiv have lower accuracy scores because I hadn't trained batchnorm layers properly.
- Attention only gives you an average of 2% boost across all architectures tested so you can delete that code if you want to save on memory/compute time.
