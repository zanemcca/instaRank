
import matplotlib
# This is needed for Mac OS
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

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
