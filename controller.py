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
    def __init__(self, target=110, gain=1, lower_bound = 75):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(
            PATIENT_PARA_FILE)

        # target BG, lower bound
        self.target = target
        self.lower_bound = lower_bound

        # controller gain
        self.gain = gain

        

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')

        action = self._bb_policy(
            pname,
            observation.CGM,
            sample_time)
        return action


    # name = patient name (for lookup)
    # glucose = cgm level in current interval
    # env_sample_time = time between samples
    def _bb_policy(self, name, glucose, env_sample_time):
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

        # scale error to units of insulin according to correction factor
        error = np.asscalar((glucose - self.target) / q.CF.values)

        # only positive correction
        if error > 0:
            bolus = error * self.gain
        else: bolus = 0

        # if bg falls below a lower bound, suspend all insulin delivery
        if glucose < self.lower_bound:
            bolus = 0
            basal = 0

        bolus = bolus / env_sample_time
        action = Action(basal=basal, bolus=bolus)
        return action
    