
import tensorflow as tf

# Helpers
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

# Evaluation Metrics
def sigmoidLossWithLogits(y_, y):
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_, y))
  return loss

def hammingLoss(y_,y):
  miss_prediction = tf.logical_xor(tf.cast(tf.round(y), tf.bool), tf.cast(tf.round(y_), tf.bool))
  return tf.reduce_mean(tf.cast(miss_prediction, tf.float32))


# Generic regression class
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
      self.hamming = hammingLoss(x, self.y)
      #self.loss = loss = sigmoidLossWithLogits(x, self.y))

      self.p_avg = tf.reduce_mean(self.p, 0)
      #self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
      self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)

      # Cross-Validation data
      xCV = data['cvX']
      yCV = data['cvY']

      scope.reuse_variables()
      xCV = getHypothesis(xCV, yCV.get_shape()[1])
      self.cv_loss = getLoss(xCV, yCV)
      self.cv_hamming = hammingLoss(xCV, yCV)

      # Test data 
      xTest = data['testX']
      yTest = data['testY']

      scope.reuse_variables()
      xTest = getHypothesis(xTest, yTest.get_shape()[1])
      self.test_loss = getLoss(xTest, yTest)



# Algorithms
def LogisticRegression(data,lr=0.1, name="logistic"):
  def getHypothesis(x, outputs):
    return linear(x, "regression/" + name, outputs)

  def getP(x):
    return sigmoid(x)

  def getLoss(x,y):
    return sigmoidLossWithLogits(x,y)

  return GenericRegression(data, getLoss, getHypothesis, getP, lr)

def MLP(data,layers=2,lr=0.1, name="mlp"):
  def getHypothesis(x, outputs):
    for i in range(layers):
      x =  tf.nn.relu(linear(x, "layer_" + str(i) + "/" + name, outputs))

    return linear(x, "regression/" + name, outputs)

  def getP(x):
    #return x
    return sigmoid(x)

  def getLoss(x,y):
    #return hammingLoss(x,y)
    return sigmoidLossWithLogits(x,y)

  return GenericRegression(data, getLoss, getHypothesis, getP, lr)

