
import tensorflow as tf
import numpy as np
import plots 

RUNS=100
BATCH_SIZE=5000
MIN_AFTER_DEQUEUE=20000

def parseFile(filename):
  files = tf.train.string_input_producer([filename])
  reader = tf.TextLineReader(skip_header_lines=1)
  key, val = reader.read(files)
  return key,val

def decodeArticles(val):
  record_defaults = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],['id'],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[1.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[1.0],[0.0],[0.0]]

  (yUp,
  yDown,
  yGetCom,
  yCreateCom,
  yGetSub,
  yCreateSub,
  itemId,
  rating,
  viewCount,
  upVoteCount,
  downVoteCount,
  getCommentsCount,
  createComment,
  notCommentRating,
  PgetComment,
  PcreateComment,
  Pup,
  Pdown,
  getSubarticlesCount,
  createSubarticle,
  notSubarticleRating,
  PgetSub,
  PcreateSub) = tf.decode_csv(val, record_defaults=record_defaults)

  features = tf.pack([
    rating,
    viewCount,
    upVoteCount,
    downVoteCount,
    getCommentsCount,
    createComment,
    notCommentRating,
    PgetComment,
    PcreateComment,
    Pup,
    Pdown,
    getSubarticlesCount,
    createSubarticle,
    notSubarticleRating,
    PgetSub,
    PcreateSub])

  y = tf.pack([
    yUp,
    yDown,
    yGetCom,
    yCreateCom,
    yGetSub,
    yCreateSub])

  return features, y

def getData():
  features, y = decodeArticles(parseFile("articleTrain.csv")[1])
  xArt, yArt = getBatch(features, y)
  features, y = decodeArticles(parseFile("articleTest.csv")[1])
  testXArt, testYArt = getBatch(features, y)
  features, y = decodeArticles(parseFile("articleCV.csv")[1])
  cvXArt, cvYArt = getBatch(features, y)
  
  article = {
    'trainX': xArt,
    'trainY': yArt,
    'testX': testXArt,
    'testY': testYArt,
    'cvX': cvXArt,
    'cvY': cvYArt
    }

  #TODO Get data for comments and subarticles as well
  #otherFiles = tf.train.string_input_producer(["subarticleTrain.csv", "subarticleTest.csv", "subarticleCV.csv", "commentTrain.csv", "commentTest.csv", "commentCV.csv"])
  return article


def getBatch(x,y):
  xBatch, yBatch = tf.train.shuffle_batch([x, y], batch_size=BATCH_SIZE, capacity=(MIN_AFTER_DEQUEUE + 3*BATCH_SIZE), min_after_dequeue=MIN_AFTER_DEQUEUE)
  return xBatch, yBatch

def linear(x, name, size, bias=True):
  w = tf.get_variable(name + "/W", [x.get_shape()[1], size])
  b = tf.get_variable(name + "/b", [1, size],
  initializer=tf.zeros_initializer)
  return tf.matmul(x, w) + b

class LogisticRegression(object):
  def __init__(self, data, lr=0.1):
    with tf.variable_scope("logisticRegression") as scope:
      self.x = x = data['trainX'] 
      #self.x = x = tf.placeholder(tf.float32, [None, 16])
      self.y = data['trainY'] 
      #self.y = tf.placeholder(tf.float32, [None, 6])
      x = linear(x, "regression", self.y.get_shape()[1])
      self.p = tf.nn.sigmoid(x)
      self.p_avg = tf.reduce_mean(self.p, 0)
      self.loss = loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x, self.y))
      self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

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


# Main function
data = getData()
model = LogisticRegression(data)

init = tf.initialize_all_variables()
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
sess.run(init)

learnPlot = plots.MultiLinePlot(linetitles=['Jtrain','Jcv'], xlimit=RUNS*BATCH_SIZE)
pPlot = plots.MultiLinePlot(linetitles=["yUp","yDown","yGetCom","yCreateCom","yGetSub","yCreateSub"], ylabel="Probability of Interaction", xlimit=RUNS*BATCH_SIZE)

# Train the model
processed = 0;
for i in range(RUNS):
  # Perform a training run
  p,_ = sess.run([model.p_avg, model.train_op])
  loss,cv_loss = sess.run([model.loss,model.cv_loss])

  processed += BATCH_SIZE
  learnPlot.addValues(processed, [loss, cv_loss])
  pPlot.addValues(processed, p)

coord.request_stop()
coord.join(threads)
