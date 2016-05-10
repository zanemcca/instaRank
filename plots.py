
import matplotlib
# This is needed for Mac OS
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class LearningPlot(object):
  def __init__(self, limit=10000):
    self.Jtrain = np.ones(1)
    self.Jcv = np.ones(1)
    self.iTrain = np.arange(1)
    self.iCV = np.arange(1)

    self.fig = plt.figure()
    ax = self.fig.add_subplot(111)

    plt.xlabel("training examples")
    plt.ylabel("error")
    plt.xlim(0,limit)
    plt.ylim(0,1)
    train,cv, = plt.plot(self.iTrain,self.Jtrain,'g',self.iCV,self.Jcv, 'r')
    self.train = train
    self.cv = cv

    self.fig.canvas.draw()
    plt.ion()
    plt.show(block=False)

  def addTrainingLoss(self, loss, samples):
    self.Jtrain = np.append(self.Jtrain, loss)
    self.iTrain = np.append(self.iTrain, samples)
    self.train.set_ydata(self.Jtrain)
    self.train.set_xdata(self.iTrain)
    self.fig.canvas.draw()

  def addCVLoss(self, loss, samples):
    self.Jcv = np.append(self.Jcv, loss)
    self.iCV = np.append(self.iCV, samples)
    self.cv.set_ydata(self.Jcv)
    self.cv.set_xdata(self.iCV)
    self.fig.canvas.draw()
