import pandas as pd
import copy
import pathos.pools
import time
import controller
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.sim_engine import SimObj
from simglucose.sensor.cgm import CGMSensor, CGMNoise
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import Action, CustomScenario
from simglucose.simulation.env import T1DSimEnv
from datetime import timedelta, datetime
import sys
import logging
logging.basicConfig(filename='MPC_Controller.log',level=logging.DEBUG,format='%(asctime)s %(message)s')


if __name__ == 'main':
    print(sys.path)

# from controller import PIDController


FRIENDLY_DATE_STR = str(datetime.strftime( datetime.now(), "%Y%m%d%H%M%S"))

def sim(sim_object):
    '''
    Simulate a sim object

    Parameters
    ----------
    sim_object: simglucose.simulation.sim_engine SimObj

    Returns
    -------
    A pandas dataframe containing the simulation results.
        axis=0: time, type datetime.datetime
        axis=1: data category, type str
    '''
    print('simulating...')
    sim_object.simulate()
    return sim_object.results()


def run_sim_PID(no_runs, patients, runtime, meals, controller_params):
    '''
    Run the simulation a single time on a list of patients with the PID controller.

    Parameters
    ----------
    no_runs: int
        the number of separate simulation runs.
    patients: list of str
        a list of patient name strings. Patient name strings can be found in the params/Quest.csv file inside simGlucose.
    runtime: int
        simulation time, in hours.
    meals: (timedelta, int)
        a tuple containing the time of meal (as referenced from simulation start) and the meal size, in grams.
    targetBG: int
        the target blood glucose for the controller, in mg/dl
    lowBG: int
        the pump suspension glucose for the controller, in mg/dl

    Returns
    -------
    A pandas dataframe containing the simulation results.
        axis=0: time, type datetime.datetime
        axis=1: MultiIndex
            level 0: data category, type str
            level 1: patient id, type str
            level 2: run number, type int (starts at 1)
    '''
    sensor = CGMSensor.withName('Dexcom')
    pump = InsulinPump.withName('Insulet')
    scenario = CustomScenario(start_time = datetime(2020, 1, 1, 0,0,0), scenario=meals)
    sim_objs = []
    keys = []
    for run in range(0, no_runs):
        for pname in patients:
            sim_objs.append(SimObj(T1DSimEnv(T1DPatient.withName(pname), 
                                sensor, 
                                pump, 
                                copy.deepcopy(scenario)), # because random numbers.
                                controller.PIDController(controller_params, pname),
                                timedelta(hours=runtime),
                                animate=False,
                                path=None))
            keys.append((run + 1, pname))
    p_start = time.time()
    print('Running batch simulation of {} items...'.format(len(patients * no_runs)))
    p = pathos.pools.ProcessPool()
    results = p.map(sim, sim_objs)
    print('Simulation took {} seconds.'.format(time.time() - p_start))
    return pd.concat(results, axis=1, keys=keys)

def run_sim_PID_once(pname, runtime, meals, controller_params):
    '''
    Run the simulation a single time on a single patient with the PID controller.

    Parameters
    ----------
    pname: str
        patient name
    runtime: int
        simulation time, in hours.
    meals: (timedelta, int)
        a tuple containing the time of meal (as referenced from simulation start) and the meal size, in grams.
    targetBG: int
        the target blood glucose for the controller, in mg/dl
    lowBG: int
        the pump suspension glucose for the controller, in mg/dl

    Returns
    -------
    A pandas dataframe containing the simulation results.
        axis=0: time, type datetime.datetime
        axis=1: data category, type str
    '''
    sensor = CGMSensor.withName('Dexcom')
    pump = InsulinPump.withName('Insulet')
    scenario = CustomScenario(start_time = datetime(2020, 1, 1, 0,0,0), scenario=meals)
    obj = SimObj(T1DSimEnv(T1DPatient.withName(pname), 
        sensor, 
        pump, 
        scenario),
        controller.PIDController(controller_params, pname),
        timedelta(hours=runtime),
        animate=False,
        path=None)
    return sim(obj)


if __name__ == '__main__':
<<<<<<< HEAD
    adults =        ["adult#001","adult#002","adult#003","adult#004","adult#005","adult#006","adult#007","adult#008"]
    t = 36
    n = 8
    meals = [(timedelta(hours=6), 80)]
    pts = adults
    ki = float(.00000)
    while ki <= 0.00001:
        ki +=.000001
        print('kp=' + str(ki))
        #           (kp, ki, kd, target, low)
        PIDparams = (0.00025, round(ki, 7), 0, 120, 70)

        dfs = run_sim_PID(n, adults, t, meals, PIDparams)

        filename = 'dfs/' + 'p_' + str(PIDparams[0]) + ' i_' + str(PIDparams[1]) + ' d_' + str(PIDparams[2]) + ' target_' + str(PIDparams[3]) + '.bz2'
        dfs.to_pickle(filename)
=======
    pname = "adult#001"
    t = 9
    meals = [(timedelta(hours=2), 50)]
    sensor = CGMSensor.withName('Dexcom')
    pump = InsulinPump.withName('Insulet')
    scenario = CustomScenario(start_time = datetime(2020, 1, 1, 0,0,0), scenario=meals)
    keys = []
    # forward horizon
    horizon = 50
    controller_params = (140, 80, horizon)
    obj= SimObj(T1DSimEnv(T1DPatient.withName(pname), 
                        sensor, 
                        pump, 
                        copy.deepcopy(scenario)), # because random numbers.
                        controller.MPCNaive(controller_params, pname),
                        timedelta(hours=t),
                        animate=False,
                        path=None)
    keys.append((1, pname))
    p_start = time.time()
    results = sim(obj)
    print('Simulation took {} seconds.'.format(time.time() - p_start))
    dfs = results
    filename = 'mpc_test.bz2'
    dfs.to_pickle(filename)
>>>>>>> MPC-exploration
        