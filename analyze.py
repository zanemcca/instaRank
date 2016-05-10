
import tensorflow as tf
import numpy as np
import plots 
import learn
import data as Data

RUNS=100
BATCH_SIZE=5000
MIN_AFTER_DEQUEUE=20000

# Main function
data = Data.getData(batchsize=BATCH_SIZE, dequeue=MIN_AFTER_DEQUEUE)
model = learn.LogisticRegression(data)

init = tf.initialize_all_variables()
sess = tf.Session()
Data.startQueue(sess=sess)
sess.run(init)

learnPlot = plots.MultiLinePlot(linetitles=['Jtrain','Jcv'], xlimit=RUNS*BATCH_SIZE)
#pPlot = plots.MultiLinePlot(linetitles=["yUp","yDown","yGetCom","yCreateCom","yGetSub","yCreateSub"], ylabel="Probability of Interaction", xlimit=RUNS*BATCH_SIZE)
pPlot = plots.MultiLinePlot(linetitles=["yUp"], ylabel="Probability of Interaction", xlimit=RUNS*BATCH_SIZE)

# Train the model
processed = 0;
for i in range(RUNS):
  # Perform a training run
  p,_ = sess.run([model.p_avg, model.train_op])
  loss,cv_loss = sess.run([model.loss,model.cv_loss])

  processed += BATCH_SIZE
  learnPlot.addValues(processed, [loss, cv_loss])
  pPlot.addValues(processed, p)

Data.stopQueue()
