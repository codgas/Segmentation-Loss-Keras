from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import cv2

def distance(label):
    tlabel = label.astype(np.uint8) 
    dist = cv2.distanceTransform(tlabel, 
                                 cv2.DIST_L2, 
                                 0)
    """
    uncomment this if you want to normalize the distance
    """
    # dist = cv2.normalize(dist, 
    #                      dist, 
    #                      0, 1.0, 
    #                      cv2.NORM_MINMAX)    
    return dist

def calc_dist_map(seg):
    H,W,C=seg.shape
    res = np.zeros_like(seg)
    for c in range(C):    
        posmask = seg[:,:,c].astype(np.bool_)
        if posmask.any():
            negmask = ~posmask
            res[:,:,c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def multiclass_surface_loss_keras(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)
if __name__ == "__main__":
    x=tf.random.uniform([8,4,2,3])
    y=tf.random.uniform([8,4,2,3])
    res=multiclass_surface_loss_keras(x,y)
    print(res)
    