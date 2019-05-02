from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import warnings
import time, json
from collections import deque

from keras.callbacks import History
from kerasPlottingUtils import PlotGenerator

#modified from https://github.com/aleju/keras
class Plotter(History):
    # see PlotGenerator.__init__() for a description of the parameters
    def __init__(self,
                 save_to_filepath=None, show_plot_window=True,
                 linestyles=None, linestyles_first_batch=None,
                 show_regressions=True,
                 poly_forward_perc=0.1, poly_backward_perc=0.2,
                 poly_n_forward_min=5, poly_n_backward_min=10,
                 poly_degree=1):
        super(Plotter, self).__init__()
        pgen = PlotGenerator(linestyles=linestyles,
                             linestyles_first_batch=linestyles_first_batch,
                             show_regressions=show_regressions,
                             poly_forward_perc=poly_forward_perc,
                             poly_backward_perc=poly_backward_perc,
                             poly_n_forward_min=poly_n_forward_min,
                             poly_n_backward_min=poly_n_backward_min,
                             poly_degree=poly_degree,
                             show_plot_window=show_plot_window,
                             save_to_filepath=save_to_filepath)
        self.plot_generator = pgen
        self.losses = []
        self.accs = []
        self.old_batch = 0
        self.last_batch = 0

    def on_batch_end(self, batch, logs={}):
        super(Plotter, self).on_batch_end(batch, logs)
        if batch==0 and self.last_batch!=0:
            self.old_batch=self.last_batch
        batch+=self.old_batch
        if batch+1==len(self.accs):
            return
        self.last_batch = batch
        train_loss = self.totals['loss']/self.seen
        train_acc = self.totals['acc']/self.seen
        self.losses.append(train_loss)
        self.accs.append(train_acc)

        self.plot_generator.update(batch, self.losses, self.accs)
