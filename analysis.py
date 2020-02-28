import pickle
import string
import pandas as pd

# prevents us from needing to run the simulation every time we run this script.
with open("./results/df.pkl",'rb') as f:
    df = pickle.load(f)

# get each col as time-series with all patients
BG = df.unstack(level=0).BG
CGM = df.unstack(level=0).CGM
insulin = df.unstack(level=0).insulin
LBGI = df.unstack(level=0).LBGI
HBGI = df.unstack(level=0).HBGI

# BG levels
BGLevels = [0, 40, 50, 60, 70, 80, 180, 250, 300, 350, 400, float("inf")]
BGpumpSuspensionsMins = [15, 30, 60, 90, 120, 150]
BGgoodRange = [80, 140]
BGokRange = [70, 180]

# Bin BG levels for each patient
BGranges = pd.DataFrame([pd.cut(BG[patient].values, bins=BGLevels).value_counts() for patient in BG.columns.values], index=BG.columns.values)
BGgoodRange = pd.DataFrame([pd.cut(BG[patient].values, bins =BGgoodRange).value_counts() for patient in BG.columns.values], index = BG.columns.values)
BGokRange = pd.DataFrame([pd.cut(BG[patient].values, bins =BGokRange).value_counts() for patient in BG.columns.values], index = BG.columns.values)

# find max and min bg per patient
BGMaxMin = pd.DataFrame(columns = ['Max BG', 'Min BG'])
BGMaxMin['Max BG'] = BG[[c for c in BG.columns.values]].max()
BGMaxMin['Min BG'] = BG[[c for c in BG.columns.values]].min()

# categorical values
dfs = [BGgoodRange, BGokRange, BGranges]

# build a dataframe with categorical values
controllerMetricsList = []
controllerMetricsKeys = []

for df in dfs:
    for key in df.columns.values:
        controllerMetricsKeys.append(key)
    for c in df:
        # maximum 
        controllerMetricsList.append(df[c].count())

controllerMetrics = pd.DataFrame(controllerMetricsList, index=controllerMetricsKeys)
print(controllerMetrics)

