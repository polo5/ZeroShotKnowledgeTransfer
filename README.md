# Zero Shot Knowledge Transfer

This is accompanying code for the paper "Zero-shot Knowledge Transfer via Adversarial Belief Matching" [see arxiv](https://arxiv.org/abs/1905.09768)

## What is this work about? (TLDR)

![picture](images/butterfly.gif)

 Our task is to compress a large neural network (teacher) into a smaller one (student), but we assume that the data used to train the teacher is not available anymore. We thus generate pseudo points adversarially (yellow markers above) and use those to match the student (right) to the teacher (left).

## Environment
- Python 3.6
- pytorch 1.0.0 (both cpu and gpu version tested)
- tensorboard 1.7.0 (for logging, + needs tensorflow)

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
@article{Micaelli2019ZeroShotKT,
  author    = {Paul Micaelli and
               Amos J. Storkey},
  title     = {Zero-shot Knowledge Transfer via Adversarial Belief Matching},
  journal   = {CoRR},
  volume    = {abs/1905.09768},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.09768},
  archivePrefix = {arXiv},
  eprint    = {1905.09768},
  timestamp = {Wed, 29 May 2019 11:27:50 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-09768},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```

## Notes
- Version 1 and 2 on arxiv have lower accuracy scores because I hadn't trained batchnorm layers properly.
- Attention only gives you an average of 2% boost across all architectures tested so you can delete that code if you want to save on memory/compute time.
