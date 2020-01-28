from simglucose.controller.base import Controller, Action
import pandas as pd
import numpy as np
import pkg_resources

PATIENT_DATA = pd.read_csv("patientData.csv")

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

class naivePID(Controller):
    def __init__(self, patient_id, init_state):
        self.init_state = init_state
        self.patient_id = patient_id
        self.tdi = PATIENT_DATA.loc[self.patient_id:'TDI']

    def policy(self, observation, reward, done, **info):
        self.state = observation
        action = Action(basal = self.tdi / 60 / 24, bolus = 0)
        return action
    def reset(self):
        self.state = self.init_state