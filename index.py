import pymongo
from pymongo import MongoClient 
import bisect
import numpy as np

# Import and prepare the plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

Jtest = np.ones(1)
Jtrain = np.ones(1)
Jcv = np.ones(1)
iteration = np.arange(1)

fig = plt.figure()

ax = fig.add_subplot(111)

plt.xlabel("training examples")
plt.ylabel("error")
plt.xlim(0,10000)
plt.ylim(0,1)
#li,li2, = plt.plot(iteration,Jtrain,'g',iteration,Jcv, 'r')
li, = plt.plot(iteration,Jtest,'g')
fig.canvas.draw()
plt.ion()
plt.show(block=False)


# Prepare and initialize the database
host="aws-us-east-1-portal.7.dblayer.com"
port="10698"
password="couchesareabit2fly4me"

InteractionsDB = MongoClient('mongodb://owner:' + password + '@' + host + ':' + port + '/interactions?ssl=true&ssl_cert_reqs=CERT_NONE').interactions
ArticlesDB = MongoClient('mongodb://owner:' + password + '@' + host + ':' + port + '/articles?ssl=true&ssl_cert_reqs=CERT_NONE').articles

Views = InteractionsDB.view
Clicks = InteractionsDB.click
UpVotes = InteractionsDB.upVote
DownVotes = InteractionsDB.downVote
Articles = ArticlesDB.article

def getId(item):
  return item._id

print "Processing article"
articles = sorted([], key=getId)
for article in Articles.find():
  art = {
    "id": article['_id'],
    "viewCount": 0,
    "upVoteCount": 0,
    "downVoteCount": 0,
    "getSubarticlesCount": 0,
    "getCommentsCount": 0,
    "createComment": 0,
    "createSubarticle": 0,
    "notSubarticleRating": 1,
    "notCommentRating": 1,
    "PgetComment": 0,
    "PgetSub": 0,
    "PcreateComment": 0,
    "PcreateSub": 0,
    "Pup": 0,
    "Pdown": 0,
  }
  bisect.insort_left(articles, art) 
  
def findArticle(articleId):
  i = bisect.bisect_left(articles, { "id" : articleId })
  if i != len(articles):
    return articles[i]
  raise ValueError('No article found with an id equal: %s' % (articleId,))
  
print "Processing subarticle"
subarticles = sorted([], key=getId)
for subarticle in ArticlesDB.subarticle.find():
  sub = {
    "id": subarticle['_id'],
    "viewCount": 0,
    "upVoteCount": 0,
    "downVoteCount": 0,
    "getCommentsCount": 0,
    "createComment": 0,
    "notCommentRating": 1,
    "PgetComment": 0,
    "PcreateComment": 0,
    "Pup": 0,
    "Pdown": 0,
  }
  bisect.insort_left(subarticles, sub) 

def findSubarticle(subId):
  i = bisect.bisect_left(subarticles, { "id" :subId })
  if i != len(subarticles):
    return subarticles[i]
  raise ValueError('No subarticle found with an id equal: %s' % (subId,))
  
print "Processing comments"
comments = sorted([], key=getId)
for comment in ArticlesDB.comment.find():
  com = {
    "id": comment['_id'],
    "viewCount": 0,
    "upVoteCount": 0,
    "downVoteCount": 0,
    "getCommentsCount": 0,
    "createComment": 0,
    "notCommentRating": 1,
    "PgetComment": 0,
    "PcreateComment": 0,
    "Pup": 0,
    "Pdown": 0,
  }
  bisect.insort_left(comments, com) 

def findComment(comId):
  i = bisect.bisect_left(comments, { "id" :comId })
  if i != len(comments):
    return comments[i]
  raise ValueError('No comment found with an id equal: %s' % (comId,))
  
  
def find(itemType, itemId):
  if itemType == 'article':
    return findArticle(itemId)
  elif itemType == 'subarticle':
    return findSubarticle(itemId)
  elif itemType == 'comment':
    return findComment(itemId)
  else:
    raise ValueError('No item found with a type of %s' % (itemType,))


# Prepare the learning algo
print "Preparing tensorflow"
import tensorflow as tf

def linear(x, name, size, bias=True):
  w = tf.get_variable(name + "/W", [x.get_shape()[1], size])
  b = tf.get_variable(name + "/b", [1, size],
  initializer=tf.zeros_initializer)
  return tf.matmul(x, w) + b

xArticle = tf.placeholder(tf.float32, [None, 15])
yArticle = tf.nn.relu(linear(xArticle, "article", 6))
yArticle = tf.nn.sigmoid(linear(yArticle, "articleReg", 6))

xSub = tf.placeholder(tf.float32, [None, 10])
ySub = tf.nn.relu(linear(xSub, "sub", 4))
ySub = tf.nn.sigmoid(linear(ySub, "subReg", 4))

xCom = tf.placeholder(tf.float32, [None, 10])
yCom = tf.nn.relu(linear(xCom, "com", 4))
yCom = tf.nn.sigmoid(linear(yCom, "comReg", 4))

_yArticle = tf.placeholder(tf.float32, [None, 6])
_ySub = tf.placeholder(tf.float32, [None, 4])
_yCom = tf.placeholder(tf.float32, [None, 4])

# We are doing multi-label classification so we want to use sigmoid with logits 
crossEntropyArticle = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(yArticle, _yArticle))
crossEntropySub = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ySub, _ySub))
crossEntropyCom = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(yCom, _yCom))

trainArticle = tf.train.AdamOptimizer(0.001).minimize(crossEntropyArticle)
trainSub = tf.train.AdamOptimizer(0.001).minimize(crossEntropySub)
trainCom = tf.train.AdamOptimizer(0.001).minimize(crossEntropyCom)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Setup the training, test and CV sets
print "Processing views"
#allviews = np.array(list(Views.find()))
#np.random.shuffle(allviews)

#cvSet = allviews[0:1001].tolist()
#testSet = allviews[1000:2001].tolist()
#trainSet = allviews[2000:].tolist()

def rateItem(item):
  # TODO Clone the javascript rating function here

def convertTrainingData():
  # TODO Return a random subset of articles for CV and Test sets
  # TODO Find all subarticles and comments belonging to the CV and tests sets
  xTrain = np.empty()
  yTrain = np.empty() 
  for view in Views.find().sort("_id", 1)
    x,y = prepTrainingSet(view)
    # TODO Seperate the views into Training, Test and Validation sets
    np.insert(xTrain, x)
    np.insert(yTrain, y)

  return (xTrain,yTrain)

def prepTrainingSet(view):
  item = find(view['viewableType'], view['viewableId'])

  if(view['viewableType'] == 'article'):
    x = np.array([[
      item['viewCount'],
      item['upVoteCount'],
      item['downVoteCount'],
      item['getCommentsCount'],
      item['createComment'],
      item['notCommentRating'],
      item['PgetComment'],
      item['PcreateComment'],
      item['Pup'],
      item['Pdown'],
      item['getSubarticlesCount'],
      item['createSubarticle'],
      item['notSubarticleRating'],
      item['PgetSub'],
      item['PcreateSub']
    ]])
    y = np.zeros((1,6))
  else:
    x = np.array([[
      item['viewCount'],
      item['upVoteCount'],
      item['downVoteCount'],
      item['getCommentsCount'],
      item['createComment'],
      item['notCommentRating'],
      item['PgetComment'],
      item['PcreateComment'],
      item['Pup'],
      item['Pdown']
    ]])
    y = np.zeros((1,4))

  item['viewCount'] = item['viewCount'] + 1

  clicks = Clicks.find({"viewId": view['_id']})
  downVotes = DownVotes.find({"viewId": view['_id']})
  upVotes = UpVotes.find({"viewId": view['_id']})
  
  # Find the labels for the session output (aka find y vector)
  # y indices => upvoted, downvoted, getCom, createCom, getSub, createSub 
  if upVotes.count() > 0:
    y[0,0] = 1
    item['upVoteCount'] = item['upVoteCount'] + 1
  if downVotes.count() > 0:
    y[0,1] = 1
    item['downVoteCount'] = item['downVoteCount'] + 1
  for click in clicks:
    if(click['type'] == "getSubarticles"):
      y[0,4] = 1 
      item['getSubarticlesCount'] = item['getSubarticlesCount'] + 1
    elif(click['type'] == "getComments"):
      y[0,2] = 1
      item['getCommentsCount'] = item['getCommentsCount'] + 1
    elif(click['type'] == "createSubarticle"):
      y[0,5] = 1 
      item['createSubarticle'] = item['createSubarticle'] + 1
    elif(click['type'] == "createComment"):
      y[0,3] = 1
      item['createComment'] = item['createComment'] + 1

  rateItem(item)

  return x,y

def trainStep(view):
  x, y = prepTrainingSet(view)

  item = find(view['viewableType'], view['viewableId'])
  if(view['viewableType'] == 'article'):
    sess.run(trainArticle, feed_dict={xArticle: x, _yArticle: y})
#    P = sess.run(yArticle, feed_dict={xArticle: x, _yArticle: y})
#    item['Pup'] = P[0,0]
#    item['Pdown'] = P[0,1]
#    item['PgetComment'] = P[0,2]
#    item['PcreateComment'] = P[0,3]
#    item['PgetSubarticle'] = P[0,4]
#    item['PcreateSubarticle'] = P[0,5]
    #print("%f (Article)" % sess.run(crossEntropyArticle, feed_dict={xArticle: x, _yArticle: y}))
  elif(view['viewableType'] == 'subarticle'):
    sess.run(trainSub, feed_dict={xSub: x, _ySub: y})
#    P = sess.run(ySub, feed_dict={xSub: x, _ySub: y})
#    item['Pup'] = P[0,0]
#    item['Pdown'] = P[0,1]
#    item['PgetComment'] = P[0,2]
#    item['PcreateComment'] = P[0,3]
    #print("%f (Subarticle)" % sess.run(crossEntropySub, feed_dict={xSub: x, _ySub: y}))
    #TODO Update the parent articles subarticleRating
  elif(view['viewableType'] == 'comment'):
    sess.run(trainCom, feed_dict={xCom: x, _yCom: y})
#    P = sess.run(yCom, feed_dict={xCom: x, _yCom: y})
#    item['Pup'] = P[0,0]
#    item['Pdown'] = P[0,1]
#    item['PgetComment'] = P[0,2]
#    item['PcreateComment'] = P[0,3]
    #print("%f (Comment)" % sess.run(crossEntropyCom, feed_dict={xCom: x, _yCom: y}))
    #TODO Update the parents notCommentRating
  else:
    print("Unknown viewableType %s" % view['viewableType'])
  
  return
    
def evaluate(views):
  print "Evaluating on %d views" % len(views)
  loss = {
      "article": 0,
      "subarticle": 0,
      "comment": 0
      }
  count = {
      "article": 0,
      "subarticle": 0,
      "comment": 0
      }
  for view in views:
    x,y = prepTrainingSet(view)
    if(view['viewableType'] == 'article'):
      loss['article'] = loss['article'] + sess.run(crossEntropyArticle, feed_dict={xArticle: x, _yArticle: y})
      count['article'] = count['article'] + 1
    elif(view['viewableType'] == 'subarticle'):
      loss['subarticle'] = loss['subarticle'] + sess.run(crossEntropySub, feed_dict={xSub: x, _ySub: y})
      count['subarticle'] = count['subarticle'] + 1
    elif(view['viewableType'] == 'comment'):
      loss['comment'] = loss['comment'] + sess.run(crossEntropyCom, feed_dict={xCom: x, _yCom: y})
      count['comment'] = count['comment'] + 1
    else:
      print("Unknown viewableType %s" % view['viewableType'])

  loss['avg'] = (loss['article'] + loss['subarticle'] + loss['comment'])/(count['article'] + count['subarticle'] + count['comment']) 
  loss['article'] = loss['article']/count['article']
  loss['subarticle'] = loss['subarticle']/count['subarticle']
  loss['comment'] = loss['comment']/count['comment']

  return loss
      

print "Converting dataset"
x,y = convertTrainingData()

# Begin training
print "Beginning Training"
processed = 0
for view in trainSet:
  trainStep(view)
  if(processed % 300 == 0):
    loss = evaluate(testSet[0:100])
    print("artLoss: {}\tsubLoss: {}\tcomLoss: {}").format(loss['article'],loss['subarticle'],loss['comment'])
    Jtest = np.append(Jtest, loss['avg'])
    iteration = np.append(iteration, processed)

    li.set_ydata(Jtest)
    li.set_xdata(iteration)
    #li2.set_ydata(Jcv)
    #li2.set_xdata(iteration)
    fig.canvas.draw()

  processed = processed + 1

