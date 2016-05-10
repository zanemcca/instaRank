
import matplotlib
# This is needed for Mac OS
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class LearningPlot(object):
  def __init__(self, limit=10000):
    # TODO implement this using the multilineplot class
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

class MultiLinePlot(object):
  def __init__(self, linetitles=[], xlimit=10000, xlabel="training examples", ylabel="error"):
    self.linetitles = linetitles
    self.lines = {}
    self.fig = plt.figure()
    ax = self.fig.add_subplot(111)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0,xlimit)
    plt.ylim(0,1)

    class Line(object):
      def __init__(self):
        self.x  = np.arange(1)
        self.y = np.zeros(1)
        self.line, = plt.plot(self.x,self.y)

      def addValue(self, x, y):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.line.set_ydata(self.y)
        self.line.set_xdata(self.x)

    for line in linetitles:
      self.lines[line] = Line()

    plt.legend(linetitles)
    self.fig.canvas.draw()
    plt.ion()
    plt.show(block=False)

  def addValue(self, linetitle, x, y):
    self.lines[linetitle].addValue(x,y)
    self.fig.canvas.draw()

  def addValues(self, x, Y):
    if len(self.linetitles) != len(Y):
      print "Length mismatch! Expected {} but len(Y) = {}".format(len(self.linetitles), len(Y))
      return

    for i in range(len(Y)):
      linetitle = self.linetitles[i]
      y = Y[i]
      self.lines[linetitle].addValue(x,y)

    self.fig.canvas.draw()
