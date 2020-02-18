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


class PController(Controller):
    def __init__(self, target=140, gain=1, pweight = 1, dweight=1.4, lower_bound = 75):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(
            PATIENT_PARA_FILE)

        # target BG, lower bound
        self.target = target
        self.lower_bound = lower_bound

        # to begin, values of bg used to calculate dxdt are set to target BG
        self.prev1 = target
        self.prev2 = target

        # controller gain, weights
        self.gain = gain
        self.pweight = pweight / (pweight + dweight)
        self.dweight = dweight / (pweight + dweight)

        

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
        # if a  patient exists fetch their data 
        if any(self.quest.Name.str.match(name)):
            q = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = np.asscalar(params.u2ss.values)
            BW = np.asscalar(params.BW.values)
        else:
            q = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                             columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43
            BW = 57.0

        # set basal rate from patient data
        # TODO: What does u2ss mean?
        basal = u2ss * BW / 6000

        '''proportional term'''
        # scale error to units of insulin according to correction factor
        pterm = np.asscalar((glucose - self.target) / q.CF.values)

        '''derivative term'''
        # unit is mg/dl per minute
        dterm = (glucose + prev1 + prev2) / 3 / env_sample_time / q.CF.values

        '''gain'''
        bolus = (pterm * self.pweight + dterm * self.dweight) * self.gain

        # cannot have negative bolus
        if bolus < 0:
            bolus = 0

        # if bg falls below a lower bound, suspend all insulin delivery
        if glucose < self.lower_bound:
            bolus = 0
            basal = 0

        bolus = bolus / env_sample_time
        action = Action(basal=basal, bolus=bolus)
        return action