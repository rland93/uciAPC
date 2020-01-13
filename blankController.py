from simglucose.patient.t1dpatient import T1DPatient
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor, CGMNoise
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Controller, Action
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim

from datetime import timedelta, datetime

import sys, inspect

'''
CREATE CONTROLLER LOGIC
'''
class blankController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state
    def policy(self, observation, reward, done, **info):
        self.state = observation
        action = Action(basal=0, bolus=0)
        return action
    def reset(self):
        self.state = self.init_state

'''
Everything following is environment setup.
'''
RANDOM_SEED = 25
# System params
results_path = './results/'
# Simulation Environment
simtime =  timedelta(hours=12)
start_time = datetime.now()
patientA = T1DPatient.withID(11)
patientB = T1DPatient.withID(1)
pump = InsulinPump.withName('Insulet')
sensor = CGMSensor.withName('Dexcom')
scenario = RandomScenario(start_time = start_time, seed = RANDOM_SEED)
environment = T1DSimEnv(patientA, sensor, pump, scenario)
# controller
controller = blankController(0)
s = SimObj(
    environment,
    controller, 
    simtime, 
    animate = False, 
    path= results_path)
results = sim(s)
print(results)