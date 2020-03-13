import pickle
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# prevents us from needing to run the simulation every time we run this script.
with open("./df.pkl",'rb') as f:
    df = pickle.load(f)

HR_INTERVAL = 2
##############  Data    ###########################
# get each col as time-series with all patients
BG_raw = df.unstack(level=0).BG.transpose()
CGM_raw = df.unstack(level=0).CGM.transpose()
insulin_raw = df.unstack(level=0).insulin.transpose()
LBGI_raw = df.unstack(level=0).LBGI.transpose()
HBGI_raw = df.unstack(level=0).HBGI.transpose()

print(BG_raw)
# construct bg ts
nstd = 1
BG = pd.DataFrame( {
    "Mean" : BG_raw.mean(),
    "Max" : BG_raw.max(),
    "Min" : BG_raw.min(),
    "Upper Envelope" : BG_raw.mean() + nstd * BG_raw.std(),
    "Lower Envelope" : BG_raw.mean() - nstd * BG_raw.std()})
BG.index.to_pydatetime()
t = BG.index

# construct insulin ts
ins = pd.DataFrame( {
    "Mean" : insulin_raw.mean(),
    "Max" : insulin_raw.max(),
    "Min" : insulin_raw.min(),
    "Upper Envelope" : insulin_raw.mean() + nstd * insulin_raw.std(),
    "Lower Envelope" : insulin_raw.mean() - nstd * insulin_raw.std()})

################### Plotting    ##################
# Grid/Subplot setup
fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(7,1)
plt.subplots_adjust(
    hspace=1,
    left=.05,
    right=.95,
    top=.95,
    bottom=.05
    )

# BG Plot
BG_ax = plt.subplot(gs[:4, :])
BG_ax.fill_between(t, BG["Upper Envelope"], BG["Lower Envelope"], alpha=0.5, color='lightblue', label="std")
BG_ax.plot(t, BG["Mean"], label="Mean", color='black')
BG_ax.plot(t, BG["Max"],linestyle='--', color='blue', label='Max')
BG_ax.plot(t, BG["Min"], linestyle='--', color='darkblue', label='Min')
BG_ax.axhspan(ymin=70, ymax=200, color='green', alpha=0.08)
BG_ax.grid(axis='x', which='both')
BG_ax.grid(axis='y', which='major')
BG_ax.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
BG_ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
BG_ax.xaxis.set_major_locator(mdates.DayLocator())
BG_ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
BG_ax.set_ylabel("Blood Glucose (mg/dl)")
BG_ax.legend()

# insulin plot
ins_ax = plt.subplot(gs[4:6, :])
# ins_ax.fill_between(t, ins["Upper Envelope"], ins["Lower Envelope"], alpha=0.5, color='lightblue', label="std")
ins_ax.step(t, ins["Mean"])
ins_ax.plot(t, ins["Max"],linestyle='--', color='blue', label='Max')
ins_ax.plot(t, ins["Min"], linestyle='--', color='darkblue', label='Min')
ins_ax.set_ylabel("Insulin (Units)")
ins_ax.grid(axis='x', which='both')
ins_ax.grid(axis='y', which='both')
ins_ax.xaxis.set_minor_locator(mdates.HourLocator(interval=HR_INTERVAL))
ins_ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
ins_ax.xaxis.set_major_locator(mdates.DayLocator())
ins_ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
ins_ax.legend()
plt.show()

# BG levels bin
BGLevels = [0, 40, 50, 60, 70, 80, 180, 250, 300, 350, 400, float("inf")]

# Bin BG levels 
BGranges = pd.DataFrame([pd.cut(BG[patient].values, bins=BGLevels).value_counts() for patient in BG.columns.values], index=BG.columns.values)

# find max and min bg per patient
BGMaxMin = pd.DataFrame(columns = ['Max BG', 'Min BG', 'Mean BG'])
BGMaxMin['Max BG'] = BG[[c for c in BG.columns.values]].max()
BGMaxMin['Min BG'] = BG[[c for c in BG.columns.values]].min()
BGMaxMin['Mean BG'] = BG[[c for c in BG.columns.values]].mean()
