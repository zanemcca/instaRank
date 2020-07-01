
import tensorflow as tf
import numpy as np
import plots 
import learn
import data as Data

RUNS=300
BATCH_SIZE=5000
MIN_AFTER_DEQUEUE=50000

# Learning Curves 
data = Data.getStaticData()

limit=10000 
learnPlot = plots.MultiLinePlot(linetitles=['MissTrain','MissCV','Jtrain','Jcv'], ylimit=2, xlimit=limit)
pPlot = plots.MultiLinePlot(linetitles=["yUp","yDown","yGetCom","yCreateCom","yGetSub","yCreateSub"], ylimit=1, ylabel="Probability of Interaction", xlimit=limit)

batches = [10, 30, 100, 300, 1000, 3000, 5000, 10000];

cv_xs = data['article']['cvX'] 
cv_ys = data['article']['cvY'] 
test_xs = data['article']['testX'][0:10]
test_ys = data['article']['testY'][0:10]

# Train the neural network for 1000 training steps
for i in batches:
  try:
    # Choose the learning algorithm
    #model = learn.LogisticRegression(data['article'], lr=3.0, static=True, pos_weight=100)
    model = learn.MLP(data['article'], lr=0.3, layers=2, static=True, pos_weight=100)

    # Initialize the session
    init = tf.initialize_all_variables()
    sess = tf.Session()

    # Old Pup calculation for comparison
    #views,upvotes = tf.split(1,2,model.x)
    #pOld = tf.reduce_mean(tf.mul(upvotes, tf.inv(tf.add(1.0,views))))

    sess.run(init)

    batch_xs = data['article']['trainX'][0:i] 
    batch_ys = data['article']['trainY'][0:i] 
    for j in range(RUNS):
      sess.run(model.train_op, feed_dict={model.x: batch_xs, model.y:  batch_ys})

    hamming,loss = sess.run([model.hamming,model.loss], feed_dict={model.x: batch_xs, model.y:  batch_ys})
    cv_hamming,cv_loss = sess.run([model.hamming, model.loss], feed_dict={model.x: cv_xs, model.y:  cv_ys})

    p = sess.run(model.p, feed_dict={model.x: test_xs, model.y: test_ys}) 

#    print np.std(p, axis=0)
    print hamming, cv_hamming,loss, cv_loss

    learnPlot.addValues(i, [hamming, cv_hamming,loss, cv_loss])

    pPlot.addValues(i,np.average(p, axis=0))

  except KeyboardInterrupt:
    break


input("Press Enter to continue...")




# Run training

# Main function
data = Data.getData(batchsize=BATCH_SIZE, dequeue=MIN_AFTER_DEQUEUE)

# Choose the learning algorithm
#model = learn.LogisticRegression(data, lr=3.0)
model = learn.MLP(data, lr=0.3, layers=2)

# Initialize the session
init = tf.initialize_all_variables()
sess = tf.Session()
saver = tf.train.Saver()

# Old Pup calculation for comparison
#views,upvotes = tf.split(1,2,model.x)
#pOld = tf.reduce_mean(tf.mul(upvotes, tf.inv(tf.add(1.0,views))))

####
# All setup must be done before the queue is started
####
Data.startQueue(sess=sess)
sess.run(init)

learnPlot = plots.MultiLinePlot(linetitles=['Jtrain','Jcv'], ylimit=0.4, xlimit=RUNS*BATCH_SIZE)
pPlot = plots.MultiLinePlot(linetitles=["yUp","yDown","yGetCom","yCreateCom","yGetSub","yCreateSub"], ylimit=0.3, ylabel="Probability of Interaction", xlimit=RUNS*BATCH_SIZE)
#pPlot = plots.MultiLinePlot(linetitles=["yUp" , "Pold"], ylabel="Probability of Interaction", xlimit=RUNS*BATCH_SIZE)

# Train the model
processed = 0;
for i in range(RUNS):
  # Perform a training run
  # TODO We need to do a static read of data inorder to plot the learning curves
  #loss,cv_loss,p,pred,_ = sess.run([model.loss,model.cv_loss,model.p_avg,pOld,model.train_op])
  loss,cv_loss,p,_ = sess.run([model.loss,model.cv_loss,model.p_avg,model.train_op])

  path = saver.save(sess, "/tmp/logisiticRegression.ckpt")

  processed += BATCH_SIZE
  learnPlot.addValues(processed, [loss, cv_loss])

  #probs = np.append(p,pred)
  #pPlot.addValues(processed, probs)
  #pPlot.addValue("yUp", processed, p)
  pPlot.addValues(processed,p)


loss,cv_loss = sess.run([model.loss,model.cv_loss])
Data.stopQueue()
print loss, cv_loss

input("Press Enter to continue...")
