from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.sensor.cgm import CGMSensor, CGMNoise
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.env import T1DSimEnv
from controller import naivePI
from datetime import timedelta, datetime

# patient setup
patientID = 12
patient = T1DPatient.withID(12)
sim_sensor = CGMSensor.withName('Dexcom')
sim_pump = InsulinPump.withName('Insulet')

# env setup
RANDOM_SEED = 25
sim_start_time = datetime.now()
sim_run_time =  timedelta(hours=24)
sim_scenario = RandomScenario(start_time = sim_start_time, seed = RANDOM_SEED)
environment = T1DSimEnv(patient, sim_sensor, sim_pump, sim_scenario)
controller = naivePI(0)

# script saves csv(s) into this path
results_path = './results/'
simulator = SimObj(
    environment,
    controller,
    sim_run_time,
    animate=False,
    path = results_path
sim(simulator)