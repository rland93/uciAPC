from simglucose.patient.t1dpatient import T1DPatient
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor, CGMNoise
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim,batch_sim
from datetime import timedelta, datetime
from multiprocessing import Process,Queue, current_process, freeze_support
import sys, argparse
from controller import blankController, naivePI

# script saves csv(s) into this path
results_path = './results/'

# set random seed for reproducibility
RANDOM_SEED = 25

# commands description
DESCRIPTION = '''Simulate patients by ID or batch, given a lower and upper range (inclusive). 
Patient ID is an int 1-30, 1-10 are adolescents, 11-20 are adults, 21-30 are children. 
use --view flag to animate the simulation. Using --view will significantly slow down the simulation.'''

# sim constants
DEFAULT_RT = 6
DEF_PUMP = 'Insulet'
DEF_CGM = 'Dexcom'

# environment setup
sim_start_time = datetime.now()
sim_scenario = RandomScenario(start_time = sim_start_time, seed = RANDOM_SEED)
sim_pump = InsulinPump.withName(DEF_PUMP)
sim_sensor = CGMSensor.withName(DEF_CGM)

# parser
parser = argparse.ArgumentParser(description=DESCRIPTION)

# patient handling
parser.add_argument('--batch', help='option to batch patients. If selected you must specify an upper and lower range with -upper and -lower.', action='store_true',)
parser.add_argument('-patient', type = int, help='patientID (number 1-30)')
parser.add_argument('-lower', type=int, help='lower ID of batch range')
parser.add_argument('-upper', type=int, help='upper ID of batch range')
# environment handling
parser.add_argument('-rt', type=int, help='run time in hours. 6 hours if no arg given.', nargs='?')
parser.add_argument('--view', help="animate simulation", action='store_true')
args = parser.parse_args()

# default runtime 6 hours, else use input
rt = timedelta(hours=DEFAULT_RT)
if args.rt:
    rt = timedelta(hours=args.rt)

# if batch is not specified, single patient; if batch is specified, lower and upper
if not args.batch:
    single_patient = int(args.patient)
elif args.batch and (args.lower is None or args.upper is None):
    parser.error("--batch requires -lower patientID and -upper patientID")
elif args.batch and args.lower and args.upper:
    patient_lower = int(args.lower)
    patient_upper = int(args.upper)
else:
    parser.error("something went wrong. Type -h for help.")

# batch
if args.batch:
    simObjs = []
    for patientid in range(patient_lower, patient_upper + 1):
        patient = T1DPatient.withID(patientid)
        environment = T1DSimEnv(patient, sim_sensor, sim_pump, sim_scenario)
        controller = naivePI(0, patientid)
        simObjs.append(SimObj(
            environment,
            controller,
            rt,
            animate=args.view,
            path = results_path
        ))
        print("Simulating patient {}.".format(str(patientid)))
    batch_sim(simObjs)

# single patient
else:
    patient = T1DPatient.withID(single_patient)
    environment = T1DSimEnv(patient, sim_sensor, sim_pump, sim_scenario)
    controller = naivePI(0, single_patient)
    simulation_obj = SimObj(environment, 
        controller, 
        rt, 
        animate=args.view, 
        path=results_path)
    sim(simulation_obj)