from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from simglucose.sensor.cgm import CGMSensor, CGMNoise
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import Action, CustomScenario
from simglucose.simulation.env import T1DSimEnv
from controller import PIDController
from datetime import timedelta, datetime
import collections
import numpy as np
import matplotlib as plt
import pandas as pd
from pandas import MultiIndex
import copy
import pkg_resources

RANDOM_SEED = 92612
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')
patient_params = pd.read_csv(PATIENT_PARA_FILE)

adolescents = ["adolescent#001",
               "adolescent#002",
               "adolescent#003",
               "adolescent#004",
               "adolescent#005",
               "adolescent#006",
               "adolescent#007",
               "adolescent#008",
               "adolescent#009",
               "adolescent#010"]
children = ["child#001",
            "child#002",
            "child#003",
            "child#004",
            "child#005",
            "child#006",
            "child#007",
            "child#008",
            "child#009",
            "child#010"]
adults =   ["adult#001",
            "adult#002",
            "adult#003",
            "adult#004",
            "adult#005",
            "adult#006",
            "adult#007",
            "adult#008",
            "adult#009",
            "adult#010"]

# patients is an array of patient NAMES. These names can be found in PATIENT_PARA_FILE.
def build_envs(scenario, start_time, patients):
    def build_env(pname):
        patient = T1DPatient.withName(pname)
        sensor = CGMSensor.withName('Dexcom')
        pump = InsulinPump.withName('Insulet')
        copied_scenario = copy.deepcopy(scenario)
        env = T1DSimEnv(patient, sensor, pump, copied_scenario)
        return env
    return [build_env(patient) for patient in patients]

# run sim for multiple patients once
def run_sim_once(simtime, meals, controller, patients):
    # times
    run_time = timedelta(hours=simtime)
    start_time = datetime(2020, 1, 1, 0,0,0)
    scenario = CustomScenario(start_time = start_time, scenario=meals)
    envs = build_envs(scenario, start_time, patients)
    # must deepcopy controllers because they're dynamic
    controllers = [copy.deepcopy(controller) for _ in range(len(envs))]
    sim_instances = [SimObj(env, ctr, run_time, animate=False, path='./results') for (env, ctr) in zip(envs, controllers)]
    # run simulations
    results = batch_sim(sim_instances, parallel=False)
    # create dataframe with results from 1 sim
    return pd.concat(results, keys=[s.env.patient.name for s in sim_instances])


# returns a dataframe of a 4d array: row level 1 is run, row level 2 is param (BG, insulin, etc.), row 3 is patient name. cols are sim times.
#
# n - simulation runs, int
# simtime - how long sim should run, int
# meals - array of tuples; time and CHO amt (timedelta, int)
# controller - which controller, Controller
# patients - patient group, string array 
# TODO: Memory will be an issue the way this is set up now

def multi_run(n, simtime, meals, controller, patients):
    frames = []
    names = []
    for run in range(0, n):
        siminstance = run_sim_once(simtime, meals, controller, patients)
        # reformat a single sim instance, time as rows, BG, CGM, etc. by patient in cols
        BG =        siminstance.unstack(level=0).BG.transpose()
        CGM =       siminstance.unstack(level=0).CGM.transpose()
        CHO =       siminstance.unstack(level=0).CHO.transpose()
        insulin =   siminstance.unstack(level=0).insulin.transpose()
        LBGI =      siminstance.unstack(level=0).LBGI.transpose()
        HBGI =      siminstance.unstack(level=0).HBGI.transpose()
        Risk =      siminstance.unstack(level=0).Risk.transpose()
        siminstance_datas  = [BG, CGM, CHO, insulin, LBGI, HBGI, Risk]
        siminstance_labels = ["BG", "CGM", "CHO", "insulin", "LBGI", "HBGI", "Risk"]

        # append run and run #
        frames.append( pd.concat(siminstance_datas, keys=siminstance_labels) )
        names.append( run )

    return pd.concat(frames, keys=names)


# controller attributes
lowBG = 60
targetBG = 120

# sim attributes
simruns = 5
runtime = 24
patients = adults
meals = [(timedelta(hours=4), 80)]

# create controller
controller = PIDController(targetBG, lowBG)

# run simulation
df = multi_run(simruns, runtime, meals, controller, patients)

# dump info disk
info = pd.DataFrame({
    "runs" : simruns,
    "runtime" : runtime,
    "patients" : patients,
    "targetBG" : targetBG,
    "lowBG" : lowBG,
    "pgain" : pgain,
    "igain" : igain,
    "dgain" : dgain
})
print(df)

df.to_pickle("results/" + str(datetime.now()) + "-run.bz2")
info.to_csv("results/" + str(datetime.now()) + "-information.csv")