from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def svm_loss(labels, logits, num_classes, penalty_parameter, weight):
    """Returns the L2-SVM loss

    :param labels:
    :param logits:
    :param num_classes:
    :param penalty_parameter:
    :param weight:
    """
    regularization_loss = tf.reduce_mean(tf.square(weight))
    hinge_loss = tf.reduce_mean(
        tf.square(
            tf.maximum(
                tf.zeros([tf.shape(logits)[0], num_classes]), 1 - labels * logits
            )
        )
    )
    loss = regularization_loss + penalty_parameter * hinge_loss
    return loss
