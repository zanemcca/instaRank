
import tensorflow as tf

# Helpers
def linear(x, name, size, bias=True, reuse=True):
  if reuse:
    w = tf.get_variable(name + "/W", [x.get_shape()[1], size])
    b = tf.get_variable(name + "/b", [1, size],
      initializer=tf.zeros_initializer)
  else:
    #w = tf.Variable(name + "/W", [x.get_shape()[1], size])
    w = tf.Variable(name=(name + "/W"), initial_value=tf.zeros([x.get_shape()[1], size]))
    b = tf.Variable(name=(name + "/b"), initial_value=tf.zeros([1, size]))

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

def weightedSigmoidLossWithLogits(y_, y, pos_weight=1):
  loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_, y, pos_weight))
  return loss

def hammingLoss(y_,y):
  miss_prediction = tf.logical_xor(tf.cast(tf.round(y), tf.bool), tf.cast(tf.round(y_), tf.bool))
  return tf.reduce_mean(tf.cast(miss_prediction, tf.float32))


# Generic regression class
class GenericRegression(object):
  def __init__(self, data, getLoss, getHypothesis, getP,lr=0.01, static=False):
    with tf.variable_scope("logisticRegression") as scope:
      if static:
        inputs = data['trainX'].shape[1]
        outputs = data['trainY'].shape[1]
        self.x = x = tf.placeholder(tf.float32, [None, inputs ])
        self.y = tf.placeholder(tf.float32, [None, outputs])
      else:
        inputs = data['trainX'].get_shape()[1]
        outputs = data['trainY'].get_shape()[1]
        self.x = x = data['trainX'] 
        self.y = data['trainY'] 

      x = getHypothesis(x, outputs)
      self.p = getP(x)
      self.loss = loss = getLoss(x, self.y)
      self.hamming = hammingLoss(x, self.y)

      self.p_avg = tf.reduce_mean(self.p, 0)
      self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)

      # Cross-Validation data
      if static is False:
        if static:
          xCV = tf.placeholder(tf.float32, [None, inputs])
          yCV = tf.placeholder(tf.float32, [None, outputs])
        else:
          xCV = data['cvX']
          yCV = data['cvY']

        scope.reuse_variables()
        xCV = getHypothesis(xCV, yCV.get_shape()[1])
        self.cv_loss = getLoss(xCV, yCV)
        self.cv_hamming = hammingLoss(xCV, yCV)

        # Test data 
        if static:
          xTest = tf.placeholder(tf.float32, [None, inputs])
          yTest = tf.placeholder(tf.float32, [None, outputs])
        else:
          xTest = data['testX']
          yTest = data['testY']

        scope.reuse_variables()
        xTest = getHypothesis(xTest, outputs)
        self.test_loss = getLoss(xTest, yTest)



# Algorithms
def LogisticRegression(data,static=False, lr=0.1, pos_weight=1, name="logistic"):
  if static:
    reuse = False
  else:
    reuse = True

  def getHypothesis(x, outputs):
    return linear(x, "regression/" + name, outputs, reuse=reuse)

  def getP(x):
    return sigmoid(x)

  def getLoss(x,y):
    return weightedSigmoidLossWithLogits(x,y, pos_weight)

  return GenericRegression(data, getLoss, getHypothesis, getP, lr, static)

def MLP(data,layers=2,lr=0.1, static=False, pos_weight=1,name="mlp"):
  if static:
    reuse = False
  else:
    reuse = True

  def getHypothesis(x, outputs):
    for i in range(layers):
      x =  tf.nn.relu(linear(x, "layer_" + str(i) + "/" + name, outputs, reuse=reuse))

    return linear(x, "regression/" + name, outputs, reuse=reuse)

  def getP(x):
    #return x
    return sigmoid(x)

  def getLoss(x,y):
    return weightedSigmoidLossWithLogits(x,y, pos_weight)

  return GenericRegression(data, getLoss, getHypothesis, getP, lr, static)
