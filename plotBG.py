import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import datetime

HYPO_RANGE = 70
HYPER_RANGE = 140
HR_INTERVAL = 2

# Colors
BGCOLOR = 'black'
CGMCOLOR = 'blue'
HYPOCOLOR = 'brown'
HYPERCOLOR = 'tan'

from pd.plotting import register_matplotlib_converters
register_matplotlib_converters()

'''
Make a blood glucose plot.
'''


def group_plot(df, nstd=1, savedir='./results', show=True):
    # data
    BG = df.unstack(level=0).BG
    t = BG.index.values

    mean_curve = BG.transpose().mean()
    std_curve = BG.transpose().std()
    up_env = mean_curve + nstd * std_curve
    down_env = mean_curve + nstd * std_curve

    # figure setup
    fig = plt.figure(figsize=(20,10))
    bgplot = plt.subplot(111)

    bgplot.fill_between(t, up_env, down_env, alpha=0.5)
    for p in BG:
        bgplot.plot(t, BG[p], '-', color='grey', alpha=0.5, label='_nolegend_')

    bgplot.plot(t, mean_curve, '-')

    bgplot.ymin = 30
    bgplot.ymax = 350
    bgplot.axhspan(ymin=HYPO_RANGE, ymax=HYPER_RANGE, color='green', alpha=0.08)
    bgplot.grid(axis='x', which='both')
    bgplot.grid(axis='y', which='major')
    bgplot.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
    bgplot.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    bgplot.xaxis.set_major_locator(mdates.DayLocator())
    bgplot.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    bgplot.set_ylabel("Blood Glucose (mg/dl)")
    bgplot.legend()
    if savedir:
        fig_path = savedir + '/figure-' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.png'
        plt.savefig(fname=fig_path)
    if show:
        plt.show()

def individual_plot(df, siminfo, savedir='./results', show=False):
    # Parse data
    df.index.to_pydatetime()
    BG = df.unstack(level=0).BG
    t = df.index.values
    CGM = df.unstack(level=0).CGM
    CHO = df.unstack(level=0).CHO
    INS = df.unstack(level=0).insulin


    # Grid/Subplot setup
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(5,1)
    plt.subplots_adjust(
        hspace=1,
        left=.05,
        right=.95,
        top=.95,
        bottom=.05
        )
    ax1 = plt.subplot(gs[:3, :])

    # ax2 = plt.subplot(gs[3,:])
    # ax3 = plt.subplot(gs[4,:])

    
    # Blood glucose plot
    mean_curve = BG.transpose().mean()
    std_curve = BG.transpose().std()
    up_env = mean_curve + nstd * std_curve
    down_env = mean_curve + nstd * std_curve

    ax1.fill_between(t, up_env, down_env, alpha=0.5)
    for p in BG:
        ax1.plot(t, BG[p], '-', color='grey', alpha=0.5, label='_nolegend_')


    ax1.plot(t, BG, label='True BG', color=BGCOLOR)
    ax1.ymin = 30
    ax1.ymax = 350
    ax1.axhspan(ymin=HYPO_RANGE, ymax=HYPER_RANGE, color='green', alpha=0.08)
    ax1.grid(axis='x', which='both')
    ax1.grid(axis='y', which='major')
    ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    ax1.set_ylabel("Blood Glucose (mg/dl)")
    ax1.legend()

'''
    # Insulin Delivered Plot
    ax2.step(t, INS)
    ax2.set_ylabel("Insulin (Units)")
    ax2.grid(axis='x', which='both')
    ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
    ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    # CHO Plot
    ax3.step(t, CHO)
    ax3.set_ylabel("CHO (g)")
    ax3.grid(axis='x', which='both')
    ax3.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
    ax3.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    if savedir:
        fig_path = savedir + '/figure-' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.png'
        plt.savefig(fname=fig_path)
    if show:
        plt.show()
'''