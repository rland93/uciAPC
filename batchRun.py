from simglucose.patient.t1dpatient import T1DPatient
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor, CGMNoise
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim,batch_sim
from datetime import timedelta, datetime
from multiprocessing import Process,Queue, current_process, freeze_support
import sys, argparse
from controller import blankController

# script saves csv(s) into this path
results_path = './results/'

# set random seed for reproducibility
RANDOM_SEED = 25

# add args
DESCRIPTION = '''batch simulate patients given a lower and upper range (inclusive). 
Patient ID is an int 1-30, 1-10 are adolescents, 11-20 are adults, 21-30 are children. 
use --view flag to animate the simulation. Using --view will significantly slow down the simulation.'''
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('patient_lower', type=int, help='lower ID of range')
parser.add_argument('patient_upper', type=int, help='upper ID of range')
parser.add_argument("--view", help="animate simulation", action='store_true')
args = parser.parse_args()

# environment setup
sim_start_time = datetime.now()
sim_run_time =  timedelta(hours=24)
sim_scenario = RandomScenario(start_time = sim_start_time, seed = RANDOM_SEED)
sim_pump = InsulinPump.withName('Insulet')
sim_sensor = CGMSensor.withName('Dexcom')

# create simulation objects
simObjs = []
for patientid in range(args.patient_lower, args.patient_upper + 1):
    patient = T1DPatient.withID(patientid)
    environment = T1DSimEnv(patient, sim_sensor, sim_pump, sim_scenario)
    controller = blankController(0)
    simObjs.append(SimObj(
        environment,
        controller,
        sim_run_time,
        animate=args.view,
        path = results_path
    ))
    print("Simulating patient {}.".format(str(patientid)))

# run simulation
batch_sim(simObjs)