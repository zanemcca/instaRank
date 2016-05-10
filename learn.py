
import tensorflow as tf
#import numpy as np

def linear(x, name, size, bias=True):
  w = tf.get_variable(name + "/W", [x.get_shape()[1], size])
  b = tf.get_variable(name + "/b", [1, size],
  initializer=tf.zeros_initializer)
  return tf.matmul(x, w) + b

def getLinear(x, outputs):
  x = linear(x, "regression", outputs)
  return x

def sigmoid(x):
  return tf.nn.sigmoid(x)

def sigmoidLossWithLogits(x, y):
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x, y))
  return loss

def LogisticRegression(data,lr=0.1, name="logistic"):
  def getHypothesis(x, outputs):
    return linear(x, "regression/" + name, outputs)

  return GenericRegression(data, sigmoidLossWithLogits, getHypothesis, sigmoid, lr)


class GenericRegression(object):
  def __init__(self, data, getLoss, getHypothesis, getP,lr=0.01):
    with tf.variable_scope("logisticRegression") as scope:
      self.x = x = data['trainX'] 
      #self.x = x = tf.placeholder(tf.float32, [None, 16])
      self.y = data['trainY'] 
      #self.y = tf.placeholder(tf.float32, [None, 6])


      x = getHypothesis(x, self.y.get_shape()[1])
      self.p = getP(x)
      self.loss = loss = getLoss(x, self.y)
      #self.loss = loss = sigmoidLossWithLogits(x, self.y))

      self.p_avg = tf.reduce_mean(self.p, 0)
      #self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
      self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)

      # TODO Measure Precision and Recall
      # Cross-Validation data
      xCV = data['cvX']
      yCV = data['cvY']

      scope.reuse_variables()
      xCV = getHypothesis(xCV, yCV.get_shape()[1])
      self.cv_loss = getLoss(xCV, yCV)

      # Test data 
      xTest = data['testX']
      yTest = data['testY']

      scope.reuse_variables()
      xTest = getHypothesis(xTest, yTest.get_shape()[1])
      self.test_loss = getLoss(xTest, yTest)








# TODO  Delete this once we perfect the above approach
class LLogisticRegression(object):
  def __init__(self, data, lr=0.01):
    with tf.variable_scope("logisticRegression") as scope:
      self.x = x = data['trainX'] 
      #self.x = x = tf.placeholder(tf.float32, [None, 16])
      self.y = data['trainY'] 
      #self.y = tf.placeholder(tf.float32, [None, 6])
      x = linear(x, "regression", self.y.get_shape()[1])
      self.p = tf.nn.sigmoid(x)
      self.p_avg = tf.reduce_mean(self.p, 0)
      self.loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x, self.y))
      #self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
      self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)

      # TODO Measure Precision and Recall
      # Cross-Validation data
      xCV = data['cvX']
      yCV = data['cvY']
      scope.reuse_variables()
      xCV = linear(xCV, "regression", yCV.get_shape()[1])
      self.cv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(xCV, yCV))
      #self.cv_train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.cv_loss)

      # Test data 
      xTest = data['testX']
      yTest = data['testY']
      scope.reuse_variables()
      xTest = linear(xTest, "regression", yTest.get_shape()[1])
      self.test_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(xTest, yTest))

