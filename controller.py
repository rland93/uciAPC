from simglucose.controller.base import Controller, Action
import pkg_resources
import numpy as np
import pandas as pd

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')
PATIENT_QUEST_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/Quest.csv')

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

class naivePI(Controller):
    def __init__(self, init_state, patientID):
        self.init_state = init_state
        self.state = init_state
        self.params = pd.read_csv(PATIENT_PARA_FILE)
        self.quest = pd.read_csv(PATIENT_QUEST_FILE)
        self.patientID = patientID
        # total daily dose = patient weight in kg
        self.tdd = self.params.loc[patientID,'BW']

    def policy(self, observation, reward, done, **kwargs):
        action = Action(basal=.2, bolus=0)
        return action

