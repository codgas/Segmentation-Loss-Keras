# Segmentation-Loss-Keras
This repository contains tensorflow / keras implementations of segmentation losses 

1-Boundary loss:paper https://arxiv.org/abs/1812.07032 
  based on the original implementation https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py
  I changed the distance calculation from scipy to openCV because I found it is much more faster 
  The keras version of the original the loss doesn't work for multiclass case so I also added that, it also works for binary case as well

--I will keep addding more loss when I have time, if you have any question please start an Issue
