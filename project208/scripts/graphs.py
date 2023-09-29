import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from calendar import month_name
import numpy as np


def barplots(x,y,data,color,xlabel,ylabel,title):
    
    #print a barplot
    plot=sns.barplot(x=x , y=y, data=data, color=color)
    plot = plot.set(xlabel=xlabel , ylabel=ylabel, title=title)
    plt.show(plot)


def lineplots(tw_series, rain_series, title_, y_label):
    
    #print lineplots
    fig, axs = plt.subplots(figsize=(10, 5))
    axs.plot_date(tw_series.index, tw_series.values, xdate=True,
            linestyle='-', mec='black', markersize=5, label='tweets', alpha=0.60)
    axs.plot_date(tw_series.index, rain_series.values[:len(tw_series.index)], xdate=True,
            linestyle=':', mec='black', markersize=5, label='rain')
    axs.fill_between(tw_series.index, tw_series.values, rain_series.values[:len(tw_series.index)],
            color='yellow', alpha=0.1)
    axs.plot(tw_series.index, np.zeros(len(tw_series.index)),
            color='green', linestyle='--', alpha=0.6)
    axs.legend()
    axs.set(title=title_, xlabel='date', ylabel=y_label)
    axs.grid()
    plt.show()
