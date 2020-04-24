import string
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# path to save figures
FRIENDLY_DATE_STR = str(datetime.datetime.strftime( datetime.datetime.now(), "%Y%m%d%H%M%S"))
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



# collected timeseries
def h_ts_collected(df):
    '''
    Create a single timeseries of collected data from multiple patients/runs.

    Parameters
    ----------
    df: pandas DataFrame object
        dataframe
    
    Returns
    -------
    dictionary:
        "index" : Series of datetime objects representing the timeseries. Type: pandas Series
        "bg" : Mean, Max, Min, Upper Env, Lower Env of blood glucose over the timeseries. Type: pandas DataFrame
        "HBGI" : Mean, Max, Min, Upper Env, Lower Env of HBGI over the timeseries. Type: pandas DataFrame
        "LBGI" : Mean, Max, Min, Upper Env, Lower Env of LBGI glucose over the timeseries. Type: pandas DataFrame
    '''
    bg = pd.DataFrame( {
        "Mean" : df.loc(axis=1)[:,:,"BG"].mean(axis=1),
        "Max" : df.loc(axis=1)[:,:,"BG"].max(axis=1),
        "Min" : df.loc(axis=1)[:,:,"BG"].min(axis=1),
        "Upper Envelope" : df.loc(axis=1)[:,:,"BG"].mean(axis=1) + nstd * df.loc(axis=1)[:,:,"BG"].std(axis=1),
        "Lower Envelope" : df.loc(axis=1)[:,:,"BG"].mean(axis=1) - nstd * df.loc(axis=1)[:,:,"BG"].std(axis=1)
        })
    hbgi = pd.DataFrame( {
        "Mean" : df.loc(axis=1)[:,:,"HBGI"].mean(axis=1),
        "Max" : df.loc(axis=1)[:,:,"HBGI"].max(axis=1),
        "Min" : df.loc(axis=1)[:,:,"HBGI"].min(axis=1),
        "Upper Envelope" : df.loc(axis=1)[:,:,"HBGI"].mean(axis=1) + nstd * df.loc(axis=1)[:,:,"HBGI"].std(axis=1),
        "Lower Envelope" : df.loc(axis=1)[:,:,"HBGI"].mean(axis=1) - nstd * df.loc(axis=1)[:,:,"HBGI"].std(axis=1)
        })
    lbgi = pd.DataFrame( {
        "Mean" : df.loc(axis=1)[:,:,"LBGI"].mean(axis=1),
        "Max" : df.loc(axis=1)[:,:,"LBGI"].max(axis=1),
        "Min" : df.loc(axis=1)[:,:,"LBGI"].min(axis=1),
        "Upper Envelope" : df.loc(axis=1)[:,:,"LBGI"].mean(axis=1) + nstd * df.loc(axis=1)[:,:,"LBGI"].std(axis=1),
        "Lower Envelope" : df.loc(axis=1)[:,:,"LBGI"].mean(axis=1) - nstd * df.loc(axis=1)[:,:,"LBGI"].std(axis=1)
        })

    return {"index": df.index, 
            "bg" : bg, 
            "HBGI" : hbgi,
            "LBGI" : lbgi}

def h_ts_sample(df):
    ''' 
    Parameters
    ----------
    df: pandas DataFrame object
        dataframe
    
    Returns
    -------
    pandas DataFrame
        Non-MultiIndex Dataframe of timeseries data from a single patient. 
    '''
    # get set of runs, pts in cols
    cols = df.columns.tolist()
    run = random.choice(tuple(set([i[0] for i in cols]))) # random run
    pt = random.choice(tuple(set([i[1] for i in cols]))) # random patient
    # we have to transpose before we get rid of the level
    # TODO this is an absolutely AWFUL way of doing this.
    return (df.loc(axis=1)[run, pt,:].transpose().droplevel(1).droplevel(0).transpose(), run, pt)

def h_gen_bg_bins(binsize, lower_lim, upper_lim):
    '''
    Generate an array of bins of a certain size.

    Parameters
    ----------
    binsize: int
        The bin size, in mg/dl.
    lower_lim: int
        The lower limit of the bins that are to be generated, in mg/dl. Not inclusive.
    upper_lim: int
        The upper limit of the bins that are to be generated, in mg/dl. Inclusive.

    Returns
    -------
    int
        The next bin divider.

    Examples
    --------
    >>> print([x for x in h_gen_bg_bins(10, 0, 50)])
    [0,10,20,30,40,50]
    '''
    x = lower_lim
    while x <= upper_lim:
        yield x
        x +=binsize

def rand_pt_bg_ts(df):
    '''
    Generate a blood glucose time series plot for a random patient and save to disk.

    Parameters
    ----------
    df: pandas DataFrame
        dataframe
    
    Returns
    -------
    None
    '''
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(6,2)
    plt.subplots_adjust(
        hspace=1,
        left=.05,
        right=.95,
        top=.95,
        bottom=.13)
    pt_data = h_ts_sample(df)
    bg_ts = plt.subplot(gs[:4,:])
    bg_ts.plot(pt_data[0].index, pt_data[0]['BG'], label='Blood Glucose')
    bg_ts.grid(axis='x', which='both')
    bg_ts.grid(axis='y', which='major')
    bg_ts.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
    bg_ts.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    bg_ts.xaxis.set_major_locator(mdates.DayLocator())
    bg_ts.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    bg_ts.set_ylabel('Blood Glucose (mg/dl)')
    bg_ts.axhspan(ymin=GOOD_CONTROL_LOWER, ymax=GOOD_CONTROL_UPPER, color='green', alpha=0.08)
    bg_ts.plot(pt_data[0].index, pt_data[0]['CGM'], label='Insulin')
    bg_ts.legend()
    ins_ts = plt.subplot(gs[4:6,:])
    ins_ts.plot(pt_data[0].index, pt_data[0]['insulin'])
    ins_ts.grid(axis='x', which='both')
    ins_ts.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
    ins_ts.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ins_ts.xaxis.set_major_locator(mdates.DayLocator())
    ins_ts.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    plt.savefig(
        './results/' + FRIENDLY_DATE_STR + '-sample_bg_ts.png', 
        dpi=DPI, 
        transparent=False)
    plt.close(fig)

def bg_ts(df):
    '''
    Generate a time series plot containing blood glucose (mean, +-1 std, and max/min envelopes) and mean HBGI/LBGI for collected all patients and save to disk.

    Parameters
    ----------
    df: pandas DataFrame
        dataframe
    
    Returns
    -------
    None
    '''
    # setup
    data = h_ts_collected(df)
    # bg plot
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(6,2)
    plt.subplots_adjust(
        hspace=1,
        left=.05,
        right=.95,
        top=.95,
        bottom=.13)
    #   subplot areas
    bg_ts = plt.subplot(gs[:4,:])
    #   plots
    bg_ts.fill_between(data["index"], data["bg"]["Upper Envelope"], data["bg"]["Lower Envelope"], alpha=0.08, color='blue', label="std")
    bg_ts.plot(data["index"], data["bg"]["Mean"], label="Mean", color='blue')
    bg_ts.plot(data["index"], data["bg"]["Max"],linestyle='--', color='mediumpurple', label='Max')
    bg_ts.plot(data["index"], data["bg"]["Min"], linestyle='--', color='slategrey', label='Min')
    #   grids, legend, etc.
    bg_ts.grid(axis='x', which='both')
    bg_ts.grid(axis='y', which='major')
    bg_ts.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
    bg_ts.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    bg_ts.xaxis.set_major_locator(mdates.DayLocator())
    bg_ts.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    bg_ts.set_ylabel('Blood Glucose (mg/dl)')
    bg_ts.axhspan(ymin=GOOD_CONTROL_LOWER, ymax=GOOD_CONTROL_UPPER, color='green', alpha=0.08)
    bg_ts.legend()
    # indicator plot
    #   subplot area
    indicators_ts = plt.subplot(gs[4:,:])
    #   plots
    indicators_ts.plot(data["index"], data["HBGI"]["Mean"], label="HBGI", color='steelblue')
    indicators_ts.plot(data["index"], data["LBGI"]["Mean"], label="LBGI", color='maroon')
    #   grids, legend, etc.
    indicators_ts.legend()
    indicators_ts.grid(axis='x', which='both')
    indicators_ts.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
    indicators_ts.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    indicators_ts.xaxis.set_major_locator(mdates.DayLocator())
    indicators_ts.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    indicators_ts.set_ylabel('HBGI/LBGI Indicator')
    plt.savefig(
        './results/' + FRIENDLY_DATE_STR + '-collective_bg_ts.png', 
        dpi=DPI, 
        transparent=False)
    plt.close(fig)
    
def h_get_bg_derivative(df):
    '''
    Get the bg derivative as a time series.

    Parameters
    ----------
    df: pandas DataFrame
        dataframe

    Returns
    -------
    pandas DataFrame
        Glucose derivative bg(t+1) - bg(t) indexed by time. Last value is zero.
        Index: datetime
        Columns: derivative
    '''
    data = h_ts_collected(df)
    index = len(data['bg'].index.values.tolist()) - 1
    derivs = []
    for i in range(0, index):
        dbg = data['bg']['Mean'].iloc[i+1] - data['bg']['Mean'].iloc[i]
        derivs.append(dbg/float(INTERVAL))
    # just add a zero at the end since we calculated derivative forward
    derivs.append(0) 
    return pd.DataFrame(derivs, index=data['index'], columns=['Derivative'])

def bg_deriv_hist(df, bins):
    '''
    Save a histogram of the blood glucose derivatives to disk.

    Parameters
    ----------
    df: pandas DataFrame
        dataframe
    bins: list of int
        List of integers representing bin boundaries

    Returns
    -------
    None
    '''
    data = h_get_bg_derivative(df)
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(1,1)
    #   subplot areas
    bg_deriv = plt.subplot(gs[:,:])
    bg_deriv.set_title("Glucose Derivative by Count")
    counts = pd.cut(data['Derivative'], bins=bins).value_counts(sort=False)
    x = counts.index.values.astype(str)
    y = counts/counts.sum() * 100

    bg_deriv.grid(axis='y', which='both')
    bg_deriv.bar(x,y,width=0.95)
    bg_deriv.set_title("Glucose Derivative by Count")
    bg_deriv.set_ylabel("Percentage of vals in range")
    bg_deriv.yaxis.set_major_formatter(mtick.PercentFormatter())
    bg_deriv.set_xlabel("Range")
    for label in bg_deriv.xaxis.get_ticklabels():
        label.set_rotation(90)
    plt.savefig(
        './results/' + FRIENDLY_DATE_STR + '-bg_deriv_counts.png', 
        dpi=DPI, 
        transparent=False)
    plt.close(fig)

def bg_counts(df, bins):
    '''
    Parameters
    ---------
    df: pandas DataFrame
        dataframe
    bins: list of int
        List of integers representing bin boundaries
    
    Returns
    -------
    None
    '''
    counts = pd.cut(df.loc(axis=1)[:,:,'BG'].unstack(), bins=bins).value_counts(sort=False)
    x = counts.index.values.astype(str) # labels
    y = counts / counts.sum() * 100

    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(1,1)
    bg_counts = plt.subplot(gs[:,:])
    bg_counts.grid(axis='y', which='both')
    bg_counts.bar(x,y,width=0.95)
    bg_counts.set_title("Glucose Values by Count")
    bg_counts.set_ylabel("Percentage of vals in range")
    bg_counts.yaxis.set_major_formatter(mtick.PercentFormatter())
    bg_counts.set_xlabel("Range")
    for label in bg_counts.xaxis.get_ticklabels():
        label.set_rotation(90)
    plt.savefig(
        './results/' + FRIENDLY_DATE_STR + '-bg_counts.png', 
        dpi=DPI, 
        transparent=False)
    plt.close(fig)

def h_gen_poincare_df(df, sample_delta, sample_interval):
    '''
    Generate a dataframe containing blood glucose at time t, and blood glucose at time t+1, indexed by t.
    '''
    bg_means = h_ts_collected(df)['bg']['Mean']
    no_samples = len(bg_means.index.values)
    bg_t1 = [bg_means[i+sample_delta] for i in range(0, no_samples - sample_delta)]
    bg_t = [bg_means[i] for i in range(0, no_samples - sample_delta)]
    indices = [i for i in bg_means.index.values[:(no_samples - sample_delta)]]
    delta_col_name = str('t_' + sample_delta)
    return pd.DataFrame([bg_t, bg_t1], index=indices, columns=['t_0', 't_' + str(sample_delta)])

if __name__ == '__main__':
    df = pd.read_pickle("./results/adults_1-8_x40.bz2")
    rand_pt_bg_ts(df)
    bg_ts(df)
    deriv_bins = [x for x in h_gen_bg_bins(.25, -4, 4)]
    bins = [x for x in h_gen_bg_bins(20, 20, 400)]
    bg_counts(df, bins)
    bg_deriv_hist(df, deriv_bins)