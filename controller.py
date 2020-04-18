from simglucose.controller.base import Controller, Action
import pkg_resources
import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)

# Patient Data
CONTROL_QUEST = pkg_resources.resource_filename(
    'simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

class blankController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state
    def policy(self, observation, reward, done, **info):
        self.state = observation
        action = Action(basal=.03, bolus=0)
        return action
    def reset(self):
        self.state = self.init_state

# A simple PID controller.
# target = target bg
# lower_bound = low bg boundary
# pgain = proportional gain
# igain = integral gain
# igain = derivative gain

class PIDController(Controller):
    def __init__(self, target, lower_bound):
        # patient params, for setting basal
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(
            PATIENT_PARA_FILE)

        # target BG, lower bound
        self.target = target
        self.lower_bound = lower_bound

        # to begin, values of bg used to calculate dxdt are set to target BG
        self.prev1 = target
        self.prev2 = target

        # set gains per patient


        # integral error
        self.ierror = 0


    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        action = self._policy(
            pname,
            observation.CGM,
            self.prev1, 
            self.prev2,
            sample_time)

        # store previous glucose readings for calculating derivative
        self.prev2 = self.prev1
        self.prev1 = observation.CGM
        return action

    # name = patient name (for lookup)
    # glucose = cgm level in current interval
    # env_sample_time = time between samples
    def _policy(self, name, glucose, prev1, prev2, env_sample_time):
        # increment integral error
        self.ierror += glucose - self.target
        
        # set basal rate from patient data
        basal = get_patient_basal(name)

        '''proportional term'''
        # scale error to units of insulin according to correction factor
        pterm = np.asscalar((glucose - self.target)

        '''integral term'''
        iterm = self.ierror

        '''derivative term'''
        # unit is mg/dl per minute
        dterm = (glucose + prev1 + prev2) / 3 / env_sample_time

        '''set bolus'''
        bolus = pterm * self.pgain + iterm * self.igain + dterm * self.dgain

        # cannot have negative bolus
        if bolus < 0:
            bolus = 0

        # if bg falls below a lower bound, suspend all insulin delivery
        if glucose < self.lower_bound:
            bolus = 0
            basal = 0

        bolus = bolus / env_sample_time
        return Action(basal=basal, bolus=bolus)


    #TODO: refactor this
    def get_patient_basal(self, name):
        '''get the basal'''
        if any(self.quest.Name.str.match(name)):
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            q = self.quest[self.quest.Name.str.match(name)]
            patient_basal = np.asscalar(params.u2ss.values) * np.asscalar(q.BW.values) / 6000
        return patient_basal

    def set_patient_gains(self, name):
        '''set gains by patient params'''
        if any(self.quest.Name.str.match(name)):
            q = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            # The Effect of Insulin Feedback on Closed Loop Glucose Control
            # Steil, 2011
            i_dir = np.asscalar(params.TDI.values) / np.asscalar(q.BW.values)
            self.pgain = i_dir / 135
            self.igain = self.pgain * 450
            self.dgain = self.pgain * 90