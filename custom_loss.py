# Función de pérdida de aproximación Lt_approx
import tensorflow as tf


def Lt_approx(zt, xt, mt):
    return tf.reduce_mean(tf.square(zt - xt * (1 - mt)))

def Ltotal_approx(Z, X, M):
    loss = 0
    for z, x, m in zip(Z, X, M):
        loss += Lt_approx(z, x, m)
    return loss

def Ltotal_target(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


def Ltotal(Z, X, M, y_pred, y_true, W_imprt, lambda_target):
    loss_approx = Ltotal_approx(Z, X, M)
    loss_target = Ltotal_target(y_pred, y_true)
    
    loss_approx = tf.cast(loss_approx, tf.float32)
    loss_target = tf.cast(loss_target, tf.float32)
    
    return loss_approx + lambda_target * loss_target
