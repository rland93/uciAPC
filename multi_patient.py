from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from simglucose.sensor.cgm import CGMSensor, CGMNoise
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import Action, CustomScenario
from simglucose.simulation.env import T1DSimEnv
from controller import PController
from datetime import timedelta, datetime
import collections
import numpy as np
import matplotlib as plt
import pandas as pd
import copy
import pkg_resources

RANDOM_SEED = 92612
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')
patient_params = pd.read_csv(PATIENT_PARA_FILE)

def build_envs(scenario, start_time):
    # all patients
    patients = patient_params['Name']
    def build_env(pname):
        patient = T1DPatient.withName(pname)
        sensor = CGMSensor.withName('Dexcom', seed =RANDOM_SEED)
        pump = InsulinPump.withName('Insulet')
        copied_scenario = copy.deepcopy(scenario)
        env = T1DSimEnv(patient, sensor, pump, copied_scenario)
        return env
    
    return [build_env(patient) for patient in patients]

# times
run_time = timedelta(hours=36)
start_time = datetime(2020, 1, 1, 0,0,0)

# scenario and environment
meals = [(timedelta(hours=4), 30),(timedelta(hours=10),75),(timedelta(hours=18),40)]
scenario = CustomScenario(start_time = start_time, scenario=meals)
envs = build_envs(scenario, start_time)

# controller
controller = PController(gain = 0.04, dweight=.5, pweight=1, target=120)
# copies of each controller for each patient
controllers = [copy.deepcopy(controller) for _ in range(len(envs))]

# list of sim instances
sim_instances = [SimObj(env, ctr, run_time, animate=False, path='./results') for (env, ctr) in zip(envs, controllers)]

# run simulations
results = batch_sim(sim_instances, parallel=False)

# create dataframe with all results
df = pd.concat(results, keys=[s.env.patient.name for s in sim_instances])

# pickle results
df.to_pickle('./results/df.pkl')