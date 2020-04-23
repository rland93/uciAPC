import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# path to save figures
fig_path = "./results/" + str(datetime.datetime.strftime( datetime.datetime.now(), "%Y%m%d%H%M%S"))
DPI = 300

# interval of time ticks on x-axis
HR_INTERVAL = 2

# range of good to bad BG readings
GOOD_CONTROL_LOWER = 80
GOOD_CONTROL_UPPER = 160

# no. of standard deviations to plot (BG timeseries chart)
nstd=1

# CHO is displayed as CHO per minute for whatever 
# interval we sample. In this case, our interval 
# is 3 minutes. Therefore, we multiply CHO value 
# by 3 to get the total CHO dosed in that minute.
# the simulator defines a "meal" as CHO dosed in
# a single minute; complex meals are not considered
INTERVAL = 3
CHO_COLOR = 'crimson'

# get data from pickle
df = pd.read_pickle("./results/100pt.bz2")

# generate time series dataframes
BG_ts = pd.DataFrame( {
    "Mean" : df.loc(axis=0)[:,"BG",:].mean(),
    "Max" : df.loc(axis=0)[:,"BG",:].max(),
    "Min" : df.loc(axis=0)[:,"BG",:].min(),
    "Upper Envelope" : df.loc(axis=0)[:,"BG",:].mean() + nstd * df.loc(axis=0)[:,"BG",:].std(),
    "Lower Envelope" : df.loc(axis=0)[:,"BG",:].mean() - nstd * df.loc(axis=0)[:,"BG",:].std()
    })
insulin_ts = pd.DataFrame( {
    "Mean" : df.loc(axis=0)[:,"insulin",:].mean(),
    "Max" :df.loc(axis=0)[:,"insulin",:].max(),
    "Min" : df.loc(axis=0)[:,"insulin",:].min(),
    "Upper Envelope" : df.loc(axis=0)[:,"insulin",:].mean() + nstd * df.loc(axis=0)[:,"insulin",:].std(),
    "Lower Envelope" : df.loc(axis=0)[:,"insulin",:].mean() - nstd * df.loc(axis=0)[:,"insulin",:].std(),
})
HBGI_ts = pd.DataFrame( {
    "Mean" : df.loc(axis=0)[:,"HBGI",:].mean(),
    "Max" :df.loc(axis=0)[:,"HBGI",:].max(),
    "Min" : df.loc(axis=0)[:,"HBGI",:].min(),
    "Upper Envelope" : df.loc(axis=0)[:,"HBGI",:].mean() + nstd * df.loc(axis=0)[:,"HBGI",:].std(),
    "Lower Envelope" : df.loc(axis=0)[:,"HBGI",:].mean() - nstd * df.loc(axis=0)[:,"HBGI",:].std(),
})
LBGI_ts = pd.DataFrame( {
    "Mean" : df.loc(axis=0)[:,"LBGI",:].mean(),
    "Max" :df.loc(axis=0)[:,"LBGI",:].max(),
    "Min" : df.loc(axis=0)[:,"LBGI",:].min(),
    "Upper Envelope" : df.loc(axis=0)[:,"LBGI",:].mean() + nstd * df.loc(axis=0)[:,"LBGI",:].std(),
    "Lower Envelope" : df.loc(axis=0)[:,"LBGI",:].mean() - nstd * df.loc(axis=0)[:,"LBGI",:].std(),
})
t = BG_ts.index.to_pydatetime()

# derivative of BG_ts (mg/dl per minute)
def get_bg_derivative(t, BG_ts):
    derivs = []
    for i in range(0, len(BG_ts.index)-1):
        dbg = BG_ts[i + 1] - BG_ts[i]
        derivs.append(dbg/float(INTERVAL))
    times = [x for i, x in enumerate(t) if i != 1]
    return pd.DataFrame(derivs, index=times)

# generate an array of bins of a certain size
def gen_bg_bins(binsize, lower_lim, upper_lim):
    x = lower_lim
    while x < upper_lim:
        yield x
        x +=binsize

# generate a dataframe containing blood glucose at time t, and blood glucose
# at time t+1, indexed by t.
def gen_poincare_df(t, BG_ts, sample_delta):
    return pd.DataFrame({
        "t+1" : [BG_ts[i+sample_delta] for i in range(0, len(BG_ts) - sample_delta)],
        "t" : [BG_ts[i] for i in range(0, len(BG_ts) - sample_delta)]
    }, index=[t for t in range(0, len(t)-sample_delta)])

# count the number of blood sugars within a certain bin and return df with bin, counts.
def gen_bg_bin_counts(BG_ts, BG_bins):
    BG_ranges = pd.DataFrame( [pd.cut(BG_ts[patientrun], bins=BG_bins).value_counts() for patientrun in BG_ts])
    return BG_ranges

################### Plotting  ##################
# Grid/Subplot setup
gs = gridspec.GridSpec(6,4)
plt.subplots_adjust(
    hspace=1,
    left=.05,
    right=.95,
    top=.95,
    bottom=.13
    )

def bg_plot(show=False):
    '''
    generate a bg timeseries plot and save to disk
    into fig_path.
    show - optional, default false; if true, show the plot in a tk window
    '''

    bg_ts_ax = plt.subplot(gs[:,:])
    bg_ts_ax.fill_between(t, BG_ts["Upper Envelope"], BG_ts["Lower Envelope"], alpha=0.08, color='blue', label="std")
    bg_ts_ax.plot(t, BG_ts["Mean"], label="Mean", color='black')
    bg_ts_ax.plot(t, BG_ts["Max"],linestyle='--', color='mediumpurple', label='Max')
    bg_ts_ax.plot(t, BG_ts["Min"], linestyle='--', color='palevioletred', label='Min')
    bg_ts_ax.axhspan(ymin=GOOD_CONTROL_LOWER, ymax=GOOD_CONTROL_UPPER, color='green', alpha=0.08)
    bg_ts_ax.grid(axis='x', which='both')
    bg_ts_ax.grid(axis='y', which='major')
    bg_ts_ax.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
    bg_ts_ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    bg_ts_ax.xaxis.set_major_locator(mdates.DayLocator())
    bg_ts_ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    bg_ts_ax.set_ylabel("Blood Glucose (mg/dl)")
    bg_ts_ax.legend()
    cho_axy = bg_ts_ax.twiny()
    cho_axx = cho_axy.twinx()
    # put CHO on same plot as BG_ts
    cho_axx.xaxis_date()
    cho_axx.bar(t, df.loc(axis=0)[0,"CHO",:].max(), color=CHO_COLOR, width=.005)
    cho_axx.set_ylim(top=120)
    cho_axx.yaxis.set_major_formatter(plt.NullFormatter())
    cho_axx.set_yticks([])
    cho_axy.set_xticks([])
    cho_axy.xaxis.set_major_formatter(plt.NullFormatter())
    cho_rects = cho_axx.patches
    cho_labels = [i*INTERVAL for i in df.loc(axis=0)[0,"CHO",:].max() if not np.isnan(i)]
    for rect, label in zip(cho_rects, cho_labels):
        if (label != 0):
            height = rect.get_height()
            cho_axx.text(rect.get_x() + rect.get_width() / 2, height, "CHO: " + str(label),
                    ha='center', va='bottom', color=CHO_COLOR)
    plt.savefig(
        fig_path + "BG_ts.png", 
        dpi=DPI, 
        transparent=False)
    if show:
        plt.show()



# Histogram of bin counts
def bg_counts(show=False):
    bg_counts_ax = plt.subplot(gs[:,:])
    bg_counts_ax.set_title("Glucose by Count")
    BG_counts = gen_bg_bin_counts(BG_ts, [bin for bin in gen_bg_bins(20,0,400)])
    x = BG_counts.sum(axis=0).index.values.astype(str) # category labels
    y = BG_counts.sum(axis=0).values / BG_counts.sum(axis=0).values.sum() * 100 # normalized
    bg_counts_ax.bar(x,y, width=0.95)
    for label in bg_counts_ax.xaxis.get_ticklabels():
        label.set_rotation(90)
    bg_counts_ax.set_ylabel("Percentage of vals in range")
    bg_counts_ax.set_xlabel("Range")
    plt.savefig(
        fig_path + "bg_counts.png", 
        dpi=DPI, 
        transparent=False)
    if show:
        plt.show()

# Histogram for bg derivative
def bg_deriv_hist(show=False):
    bg_deriv_hist_ax = plt.subplot()
    bg_deriv_hist_ax.set_title("Glucose Derivative by Count")
    BG_deriv_counts = gen_bg_bin_counts(get_bg_derivative(t, BG_ts["Mean"]), [bin for bin in gen_bg_bins(0.05, 0, 1)])
    x = BG_deriv_counts.sum(axis=0).index.values.astype(str)
    y = BG_deriv_counts.sum(axis=0).values
    bg_deriv_hist_ax.bar(x,y, width=0.95)
    for label in bg_deriv_hist_ax.xaxis.get_ticklabels():
        label.set_rotation(90)
    plt.savefig(
        fig_path + "bg_deriv_hist.png", 
        dpi=DPI, 
        transparent=False)
    if show:
        plt.show()
'''
# Poincare Charts
BG_poincare_ax1 = plt.subplot(gs[5:, 0:1])
BG_poincare_ax1.set_title("Random Sample #1")
BG_poincare_ax2 = plt.subplot(gs[5:, 1:2])
BG_poincare_ax2.set_title("Random Sample #2")
BG_poincare_ax3 = plt.subplot(gs[5:, 2:3])
BG_poincare_ax3.set_title("Random Sample #3")
BG_poincare_ax4 = plt.subplot(gs[5:, 3:])
BG_poincare_ax4.set_title("Random Sample #4")
sample = df.loc(axis=0)[:,"BG",:].sample(4, axis=0, replace=False).transpose()
poincares = []
for col in sample:
    poincares.append(gen_poincare_df(sample[col].index, sample[col].values, 10))
    
BG_poincare_ax1.scatter(poincares[0]["t"], poincares[0]["t+1"])
BG_poincare_ax2.scatter(poincares[1]["t"], poincares[1]["t+1"])
BG_poincare_ax3.scatter(poincares[2]["t"], poincares[2]["t+1"])
BG_poincare_ax4.scatter(poincares[3]["t"], poincares[3]["t+1"])
'''