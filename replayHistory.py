import pymongo
import sys
import progressbar
import timeit
import random
import math
import csv
from pymongo import MongoClient 
import bisect
import numpy as np
from operator import itemgetter
from SortedCollection import SortedCollection

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
Subarticles = ArticlesDB.subarticle
Comments = ArticlesDB.comment

def getId(item):
  return str(item['_id'])

def getViewId(item):
  return str(item['viewId'])

print "caching downvotes"
downVotes = SortedCollection(list(DownVotes.find().sort("viewId", 1)), key=getViewId)
#downVotes = SortedCollection(list(DownVotes.find()), key=itemgetter('viewId'))
print "caching upvotes"
upVotes = SortedCollection(list(UpVotes.find().sort("viewId", 1)), key=getViewId)
#upVotes = SortedCollection(list(UpVotes.find()), key=itemgetter('viewId'))
#upVotes = SortedCollection([], key=itemgetter('viewId'))
print "caching clicks"
clicks = SortedCollection(list(Clicks.find().sort("viewId", 1)), key=getViewId)
#clicks = SortedCollection(list(Clicks.find()), key=itemgetter('viewId'))
#clicks = SortedCollection([], key=itemgetter('viewId'))


def findGeneric(lst,Id):
  Id = str(Id)
  try: 
    item = lst.find(Id)
  except ValueError as e:
    return
  return item 

print "Processing article"
articles = SortedCollection([], key=itemgetter('id'))
for article in Articles.find():
  art = {
    "id": str(article['_id']),
    "rating": 0,
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
  articles.insert(art)
  
def findArticle(articleId):
  articleId = str(articleId)
  try: 
    art = articles.find(articleId)
  except ValueError as e:
    return
  return art
  
print "Processing subarticle"
subarticles = SortedCollection([], key=itemgetter('id'))
for subarticle in ArticlesDB.subarticle.find():
  sub = {
    "id": str(subarticle['_id']),
    "parentId": str(subarticle['parentId']),
    "rating": 0,
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
  subarticles.insert(sub)

def findSubarticle(Id):
  Id = str(Id)
  try: 
    sub = subarticles.find(Id)
  except ValueError as e:
    return
  return sub 
  
print "Processing comments"
comments = SortedCollection([], key=itemgetter('id'))
for comment in ArticlesDB.comment.find():
  com = {
    "id": str(comment['_id']),
    "commentableId": str(comment['commentableId']),
    "commentableType": comment['commentableType'],
    "rating": 0,
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
  comments.insert(com)

def findComment(Id):
  Id = str(Id)
  try: 
    com = comments.find(Id)
  except ValueError as e:
    return
  return com 

  
def find(itemType, itemId):
  if itemType == 'article':
    return findArticle(itemId)
  elif itemType == 'subarticle':
    return findSubarticle(itemId)
  elif itemType == 'comment':
    return findComment(itemId)
  else:
    raise ValueError('No item found with a type of %s' % (itemType,))


def rateItem(rateable):
  bonus = {
    'upVoteCount': 1,
    'downVoteCount': 0,
    'getCommentsCount': 1,
    'getSubarticlesCount': 1,
    'createSubarticleCount': 0,
    'createCommentCount': 0,
    'viewCount': 1 
  }

  weight = {
    'upVotes': 1,
    'downVotes': 1,
    'subarticles': 0.7, 
    'comments': 0.7
  }

  tTable = [
    1,
    0.816,
    0.765,
    0.741,
    0.727,
    0.718,
    0.711,
    0.706,
    0.703,
    0.700,
    0.697,
    0.695,
    0.694,
    0.692,
    0.691,
    0.690,
    0.689,
    0.688,
    0.688,
    0.687,
    0.687,
    0.686,
    0.686,
    0.685,
    0.685,
    0.684,
    0.684,
    0.684,
    0.683,
    0.683,
    0.683,
    0.681,
    0.679,
    0.679,
    0.678,
    0.677
  ];
  criticalValueInfinity = 0.674

  upVoteCount = float(rateable['upVoteCount'])
  downVoteCount = float(rateable['downVoteCount'])
  getCommentsCount = float(rateable['getCommentsCount'])
  viewCount = float(rateable['viewCount']) + 1

  PnotCom = 1
  if 'notCommentRating' in rateable:
    PnotCom = float(rateable['notCommentRating'])

  PnotSub = 1
  getSubarticlesCount = 0
  if 'notSubarticleRating' in rateable:
    PnotSub = float(rateable['notSubarticleRating'])
    getSubarticlesCount = float(rateable['getSubarticlesCount'])

  if(upVoteCount >= viewCount):
    print('Upvote count is too high for ' + rateable['id'] + '! Rating broken!')
    return 0.0001
  if(downVoteCount >= viewCount):
    print('Downvote count is too high for ' + rateable['id'] + '! Rating broken!')
    return 0.0001

  if 'upVoteCount' in rateable:
    upVoteCount += float(rateable['upVoteCount'])
  if 'downVoteCount' in rateable:
    downVoteCount += float(rateable['downVoteCount'])
  if 'viewCount' in rateable:
    viewCount += float(rateable['viewCount'])
  else:
    print('Rateable item does not have a viewCount!')
    print rateable['viewCount']

  viewCountBonus = viewCount + float(bonus['viewCount'])

  # Calculate the critical value from the df and tTable
  degreesOfFreedom = int(viewCountBonus - 1)
  if(degreesOfFreedom <= 0):
    criticalValue = 1
  elif(degreesOfFreedom < len(tTable)):
    criticalValue = tTable[degreesOfFreedom - 1]
  else:
    criticalValue = criticalValueInfinity

  clickCount = upVoteCount + downVoteCount 

  Pcom = 0

  clickCount += getCommentsCount
  Pcom = (1 - PnotCom)
  Pcom *= weight['comments']

  # P(click & comment interaction)
  # Q function of geometric distribution for P(getCommentsCount > 0)
  PgetComs = (getCommentsCount/(getCommentsCount + viewCount))

  # Biased Margninal Error = Critical Value * stddev/sqrt(n)
  PgetComsBonus = getCommentsCount + bonus['getCommentsCount']
  PgetComsBonus /= (PgetComsBonus + viewCountBonus)

  # The variance on the geometric series can grow indefinitely which can result in massive bonuses
  # so to avoid exploitation we are going to use the approximation of the bernoulli stderr
  PgetComsErrorBonus = criticalValue * math.sqrt(PgetComsBonus*(1-PgetComsBonus)/viewCountBonus)

  PgetComs += PgetComsErrorBonus

  Pcom *= PgetComs

  Psub = 0
  if 'notSubarticleRating' in rateable:
    clickCount += getSubarticlesCount
    Psub = (1 - PnotSub)
    Psub *= weight['subarticles']

    # P(click & subarticle interaction)
    #
    # The Bernoulli distribution is to easily broken by a getSubarticlesCount > viewCount
    # so I am using a geometric distribution instead. Also views are not recreated when you go back
    # and forth from the feed and articles so it is geometric in nature anyway
    #
    # Bernoulli distribution
    # PgetSubs = (getSubarticlesCount/viewCount)
    # Biased Margninal Error = Critical Value * stddev/sqrt(n)
    # PgetSubsBonus = getSubarticlesCount + bonus['getSubarticlesCount
    # PgetSubsBonus /= viewCountBonus
    #console.log(PgetSubBonus)

    # Q function of geometric distribution for P(clicking on article > 0 times in a view)
    PgetSubs = (getSubarticlesCount/(getSubarticlesCount + viewCount))
    # Biased Margninal Error = Critical Value * stddev/sqrt(n)
    PgetSubsBonus = getSubarticlesCount + bonus['getSubarticlesCount']
    PgetSubsBonus /= (PgetSubsBonus + viewCountBonus)

    # The variance on the geometric series can grow indefinitely which can result in massive bonuses
    PgetSubsErrorBonus = criticalValue * math.sqrt(PgetSubsBonus*(1-PgetSubsBonus)/viewCountBonus)

    PgetSubs += PgetSubsErrorBonus
    Psub *= PgetSubs

    rateable['PgetSub'] = PgetSubs
    rateable['PcreateSub'] = Psub

  Pup = upVoteCount/viewCount

  PupBonus = upVoteCount + bonus['upVoteCount']
  PupBonus /= viewCountBonus
  PupErrorBonus = criticalValue * math.sqrt(PupBonus*(1-PupBonus)/viewCountBonus) # The add the margin error (CV * stderr)
  Pup += PupErrorBonus
  Pup = weight['upVotes'] * Pup

  Pdown = weight['downVotes'] * downVoteCount/viewCount

  Pclick = Pcom + Psub - Pcom*Psub
  #rating = getUnion(Pup, Pclick) - getIntersection(Pclick, Pdown)

  # The rating is the probability of not downvoting and then positively interacting
  rating = (1 - Pdown)*(Pup + (1 - Pdown - Pup)*Pclick)

  if(rating > 1 or rating < 0 or math.isnan(rating)):
    print('The returned probability is not unitary!: ' + rating)
    return

  if(rating == 1):
    rating = 0.9999 
  if(rating == 0):
    rating = 0.0001

  rateable['PgetComment'] = PgetComs
  rateable['PcreateComment'] = Pcom
  rateable['Pup'] = Pup
  rateable['Pdown'] = Pdown
  rateable['rating'] = rating
  return

def prepTrainingSet(view):
  item = find(view['viewableType'], view['viewableId'])
  if item:
    if(view['viewableType'] == 'article'):
      x = np.array([[
        view['_id'],
        item['rating'],
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
        view['_id'],
        item['rating'],
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
    
    # Find the labels for the session output (aka find y vector)
    # y indices => upvoted, downvoted, getCom, createCom, getSub, createSub 
    if findGeneric(upVotes,view['_id']):
      y[0,0] = 1
      item['upVoteCount'] = item['upVoteCount'] + 1
    if findGeneric(downVotes, view['_id']):
      y[0,1] = 1
      item['downVoteCount'] = item['downVoteCount'] + 1

    click = findGeneric(clicks,view['_id'])
    while click:
      clicks.remove(click)
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
      click = findGeneric(clicks,view['_id'])

    oldRating = item['rating']
    rateItem(item)
    rating = item['rating']
    if view['viewableType'] == 'subarticle':
      parent = find('article', item['parentId'])
      if parent:
        parent['notSubarticleRating'] *= (1 - rating)/(1 - oldRating)
      #else: print "No parent article"
    if view['viewableType'] == 'comment':
      parent = find(item['commentableType'], item['commentableId'])
      if parent: parent['notCommentRating'] *= (1 - rating)/(1 - oldRating)
      #else: print "No parent " + item['commentableType'] 

    return x,y
  else:
    return [],[]

# Converts the views, clicks and votes into session histories by replaying history and updating the item ratings
def convertTrainingData():
  data = {
      'articleTrain.csv': [['yUp','yDown','yGetCom','yCreateCom','yGetSub','yCreateSub','id', 'rating','viewCount','upVoteCount','downVoteCount','getCommentsCount','createComment','notCommentRating','PgetComment','PcreateComment','Pup','Pdown','getSubarticlesCount','createSubarticle','notSubarticleRating','PgetSub','PcreateSub']],
      'articleTest.csv': [['yUp','yDown','yGetCom','yCreateCom','yGetSub','yCreateSub','id', 'rating','viewCount','upVoteCount','downVoteCount','getCommentsCount','createComment','notCommentRating','PgetComment','PcreateComment','Pup','Pdown','getSubarticlesCount','createSubarticle','notSubarticleRating','PgetSub','PcreateSub']],
      'articleCV.csv': [['yUp','yDown','yGetCom','yCreateCom','yGetSub','yCreateSub','id', 'rating','viewCount','upVoteCount','downVoteCount','getCommentsCount','createComment','notCommentRating','PgetComment','PcreateComment','Pup','Pdown','getSubarticlesCount','createSubarticle','notSubarticleRating','PgetSub','PcreateSub']],
      'subarticleTrain.csv': [['yUp','yDown','yGetCom','yCreateCom','yGetSub','yCreateSub','id', 'rating','viewCount','upVoteCount','downVoteCount','getCommentsCount','createComment','notCommentRating','PgetComment','PcreateComment','Pup','Pdown']],
      'subarticleTest.csv': [['yUp','yDown','yGetCom','yCreateCom','yGetSub','yCreateSub','id', 'rating','viewCount','upVoteCount','downVoteCount','getCommentsCount','createComment','notCommentRating','PgetComment','PcreateComment','Pup','Pdown']],
      'subarticleCV.csv': [['yUp','yDown','yGetCom','yCreateCom','yGetSub','yCreateSub','id', 'rating','viewCount','upVoteCount','downVoteCount','getCommentsCount','createComment','notCommentRating','PgetComment','PcreateComment','Pup','Pdown']],
      'commentTrain.csv': [['yUp','yDown','yGetCom','yCreateCom','yGetSub','yCreateSub','id', 'rating','viewCount','upVoteCount','downVoteCount','getCommentsCount','createComment','notCommentRating','PgetComment','PcreateComment','Pup','Pdown']],
      'commentTest.csv': [['yUp','yDown','yGetCom','yCreateCom','yGetSub','yCreateSub','id', 'rating','viewCount','upVoteCount','downVoteCount','getCommentsCount','createComment','notCommentRating','PgetComment','PcreateComment','Pup','Pdown']],
      'commentCV.csv': [['yUp','yDown','yGetCom','yCreateCom','yGetSub','yCreateSub','id', 'rating','viewCount','upVoteCount','downVoteCount','getCommentsCount','createComment','notCommentRating','PgetComment','PcreateComment','Pup','Pdown']]
      }

  arts = articles[:]
  random.shuffle(arts)

  cvLength = int(0.05 * len(arts))
  testLength = int(0.10 * len(arts))

  cvArts = SortedCollection(arts[0:cvLength], key=itemgetter('id'))
  testArts = SortedCollection(arts[cvLength:(cvLength + testLength)], key=itemgetter('id'))
  trainArts = SortedCollection(arts[(cvLength + testLength):], key=itemgetter('id'))

  cvSubs = SortedCollection([], key=getId)
  testSubs = SortedCollection([], key=getId)
  trainSubs = SortedCollection([], key=getId)

  cvComs = SortedCollection([], key=getId)
  testComs = SortedCollection([], key=getId)
  trainComs = SortedCollection([], key=getId)

  def contains(lst, Id):
    try: 
      if lst.find(str(Id)):
        return True
      else:
        return False
    except ValueError:
      return False
  

  for sub in Subarticles.find():
    if contains(testArts, sub['parentId']):
      testSubs.insert(sub)
    elif contains(cvArts, sub['parentId']):
      cvSubs.insert(sub)
    else:
      trainSubs.insert(sub)

  replies = []
  for comment in Comments.find():
    if comment['commentableType'] == 'article':
      if contains(cvArts, comment['commentableId']):
        cvComs.insert(comment)
      elif contains(testArts, comment['commentableId']):
        testComs.insert(comment)
      else:
        trainComs.insert(comment)
    elif comment['commentableType'] == 'subarticle':
      if contains(cvSubs, comment['commentableId']):
        cvComs.insert(comment)
      elif contains(testSubs, comment['commentableId']):
        testComs.insert(comment)
      else:
        trainComs.insert(comment)
    else:
      replies.append(comment)

  for comment in replies: 
    if comment['commentableType'] == 'article':
      if contains(cvArts, comment['commentableId']):
        cvComs.insert(comment)
      elif contains(testArts, comment['commentableId']):
        testComs.insert(comment)
      else:
        trainComs.insert(comment)
    elif comment['commentableType'] == 'subarticle':
      if contains(cvSubs, comment['commentableId']):
        cvComs.insert(comment)
      elif contains(testSubs, comment['commentableId']):
        testComs.insert(comment)
      else:
        trainComs.insert(comment)
    elif comment['commentableType'] == 'comment':
      if contains(cvComs, comment['commentableId']):
        cvComs.insert(comment)
      elif contains(testComs, comment['commentableId']):
        testComs.insert(comment)
      else:
        trainComs.insert(comment)
    else:
      print "Comment on a comment!"

  print("cvArts: {}\tcvSubs: {}\tcvComs: {}").format(len(cvArts),len(cvSubs),len(cvComs))
  print("testArts: {}\ttestSubs: {}\ttestComs: {}").format(len(testArts),len(testSubs),len(testComs))
  print("trainArts: {}\ttrainSubs: {}\ttrainComs: {}").format(len(trainArts),len(trainSubs),len(trainComs))
  sys.stdout.flush()

  views = Views.find().sort("_id",1)
  viewLength = views.count()
  pbar = progressbar.ProgressBar(widgets=[progressbar.Timer(),progressbar.ETA(), progressbar.Bar(), progressbar.Percentage()], maxval=viewLength).start()

  processed = int(0)
  for view in views:
    pbar.update(processed)
    processed = processed + 1
    x,y = prepTrainingSet(view)
    if len(x): 
      dat = np.append(y,x).tolist()
      if(view['viewableType'] == 'article'):
        if contains(cvArts, view['viewableId']):
          data['articleCV.csv'].append(dat)
        elif contains(testArts, view['viewableId']):
          data['articleTest.csv'].append(dat)
        else:
          data['articleTrain.csv'].append(dat)
      elif(view['viewableType'] == 'subarticle'):
        if contains(cvSubs, view['viewableId']):
          data['subarticleCV.csv'].append(dat)
        elif contains(testSubs, view['viewableId']):
          data['subarticleTest.csv'].append(dat)
        else:
          data['subarticleTrain.csv'].append(dat)
      elif(view['viewableType'] == 'comment'):
        if contains(cvComs, view['viewableId']):
          data['commentCV.csv'].append(dat)
        elif contains(testComs, view['viewableId']):
          data['commentTest.csv'].append(dat)
        else:
          data['commentTrain.csv'].append(dat)
      else:
        print "Unknown viewableType: {}"

  pbar.finish()
  print "Writing results"
  for filename, lst in data.items():
    with open(filename, 'wb') as csvfile:
      writer = csv.writer(csvfile)
      for line in lst:
        writer.writerow(line)



print "Converting dataset"
convertTrainingData()
