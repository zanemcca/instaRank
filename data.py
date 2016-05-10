
import tensorflow as tf

coord = tf.train.Coordinator()

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

  features = tf.pack([viewCount,upVoteCount])
#  features = tf.pack([
#    rating,
#    viewCount,
#    upVoteCount,
#    downVoteCount,
#    getCommentsCount,
#    createComment,
#    notCommentRating,
#    PgetComment,
#    PcreateComment,
#    Pup,
#    Pdown,
#    getSubarticlesCount,
#    createSubarticle,
#    notSubarticleRating,
#    PgetSub,
#    PcreateSub])

  y = tf.pack([yUp])
  #y = tf.pack([
  #  yUp,
  #  yDown,
  #  yGetCom,
  #  yCreateCom,
  #  yGetSub,
  #  yCreateSub])

  return features, y

def getData(batchsize, dequeue):
  features, y = decodeArticles(parseFile("articleTrain.csv")[1])
  xArt, yArt = getBatch(features, y, batchsize, dequeue)
  features, y = decodeArticles(parseFile("articleTest.csv")[1])
  testXArt, testYArt = getBatch(features, y, batchsize, dequeue)
  features, y = decodeArticles(parseFile("articleCV.csv")[1])
  cvXArt, cvYArt = getBatch(features, y, batchsize, dequeue)
  
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

def getBatch(x,y,batchsize, dequeue):
  xBatch, yBatch = tf.train.shuffle_batch([x, y], batch_size=batchsize, capacity=(dequeue + 3*batchsize), min_after_dequeue=dequeue)
  return xBatch, yBatch

def startQueue(sess):
  global threads
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)
  return threads

def stopQueue():
  coord.request_stop()
  coord.join(threads)
  return 
