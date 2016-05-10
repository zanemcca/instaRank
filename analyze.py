
import tensorflow as tf
import numpy as np
import plots 
import learn
import data as Data

RUNS=500
BATCH_SIZE=2000
MIN_AFTER_DEQUEUE=10000

# Main function
data = Data.getData(batchsize=BATCH_SIZE, dequeue=MIN_AFTER_DEQUEUE)

# Choose the learning algorithm
#model = learn.LogisticRegression(data)
model = learn.MLP(data, lr=0.3)

# Initialize the session
init = tf.initialize_all_variables()
sess = tf.Session()
saver = tf.train.Saver()

# Old Pup calculation for comparison
views,upvotes = tf.split(1,2,model.x)
pOld = tf.reduce_mean(tf.mul(upvotes, tf.inv(tf.add(1.0,views))))

####
# All setup must be done before the queue is started
####
Data.startQueue(sess=sess)
sess.run(init)

learnPlot = plots.MultiLinePlot(linetitles=['Jtrain','Jcv'], xlimit=RUNS*BATCH_SIZE)
#pPlot = plots.MultiLinePlot(linetitles=["yUp","yDown","yGetCom","yCreateCom","yGetSub","yCreateSub"], ylabel="Probability of Interaction", xlimit=RUNS*BATCH_SIZE)
pPlot = plots.MultiLinePlot(linetitles=["yUp" , "Pold"], ylabel="Probability of Interaction", xlimit=RUNS*BATCH_SIZE)

# Train the model
processed = 0;
for i in range(RUNS):
  # Perform a training run
  # TODO We need to do a static read of data inorder to plot the learning curves
  loss,cv_loss,p,pred,_ = sess.run([model.loss,model.cv_loss,model.p_avg,pOld,model.train_op])

  path = saver.save(sess, "/tmp/logisiticRegression.ckpt")

  processed += BATCH_SIZE
  learnPlot.addValues(processed, [loss, cv_loss])

  probs = np.append(p,pred)
  pPlot.addValues(processed, probs)


loss,cv_loss = sess.run([model.loss,model.cv_loss])
Data.stopQueue()
print loss, cv_loss

input("Press Enter to continue...")
