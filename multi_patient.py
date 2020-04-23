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





def run_sim_once(simtime, meals, patients, targetBG, lowBG):
    run_time = timedelta(hours=simtime)
    start_time = datetime(2020, 1, 1, 0,0,0)
    scenario = CustomScenario(start_time = start_time, scenario=meals)
    def build_env(pname):
        patient = T1DPatient.withName(pname)
        controller = PIDController(targetBG, lowBG, pname)
        sensor = CGMSensor.withName('Dexcom')
        pump = InsulinPump.withName('Insulet')
        copied_scenario = copy.deepcopy(scenario)
        env = T1DSimEnv(patient, sensor, pump, copied_scenario)
        instance = SimObj(env, controller, run_time, path='./results', animate = False)
        return instance
    # run batch sim
    results = batch_sim([build_env(pname) for pname in patients], parallel=True)
    return pd.concat(results)

def multi_run(n_runs, simtime, meals, patients, target, low):
    frames = []
    names = []
    for run in range(0, n_runs):
        siminstance = run_sim_once(simtime, meals, patients, target, low)
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



t = 4
n = 2
meals = [(timedelta(hours=2), 80)]
twopts = ["adult#001","adult#002"]
target = 100
low = 60


df = multi_run(n, t, meals, twopts, target, low)
print(df)