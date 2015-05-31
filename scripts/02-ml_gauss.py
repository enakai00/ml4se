# -*- coding: utf-8 -*-
#
# 最尤推定による正規分布の推定
#
# 2015/04/23 ver1.0
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import normal
from scipy.stats import norm

if __name__ == '__main__':
    fig = plt.figure()
    for c, datapoints in enumerate([2,4,10,100]): # サンプル数
        ds = normal(loc=0, scale=1, size=datapoints)
        mu = np.mean(ds)                # 平均の推定値
        sigma = np.sqrt(np.var(ds))     # 標準偏差の推定値

        subplot = fig.add_subplot(2,2,c+1)
        subplot.set_title("N=%d" % datapoints)
        # 真の曲線を表示
        linex = np.arange(-10,10.1,0.1)
        orig = norm(loc=0, scale=1)
        subplot.plot(linex, orig.pdf(linex), color='green', linestyle='--')
        # 推定した曲線を表示
        est = norm(loc=mu, scale=np.sqrt(sigma))
        label = "Sigma=%.2f" % sigma
        subplot.plot(linex, est.pdf(linex), color='red', label=label)
        subplot.legend(loc=1)
        # サンプルの表示
        subplot.scatter(ds, orig.pdf(ds), marker='o', color='blue')
        subplot.set_xlim(-4,4)
        subplot.set_ylim(0)
    fig.show()
