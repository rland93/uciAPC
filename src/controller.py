from simglucose.controller.base import Controller, Action
import pkg_resources
import numpy as np
import pandas as pd
import logging
import math
import cvxpy as cp

logger = logging.getLogger(__name__)

# Patient Data
CONTROL_QUEST = pkg_resources.resource_filename(
    'simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

class BlankController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state
    def policy(self, observation, reward, done, **info):
        self.state = observation
        action = Action(basal=.03, bolus=0)
        return action
    def reset(self):
        self.state = self.init_state

class TunedPIDController(Controller):
    def __init__(self, controller_params, name):
        # patient params, for setting basal
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(
            PATIENT_PARA_FILE)
        
        self.target = controller_params[0]
        self.lower_bound = controller_params[1]
        self.tau_c = controller_params[2]

        # to begin, values of bg used to calculate dxdt are set to target BG
        self.prev1 = self.target
        self.prev2 = self.target

        ''' basal, PID gains are set patient-to-patient'''
        if any(self.quest.Name.str.match(name)):
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            quest = self.quest[self.quest.Name.str.match(name)]
            self.patient_BW = np.asscalar(params.BW.values)
            self.patient_basal = np.asscalar(params.u2ss.values) * self.patient_BW / 6000
            self.patient_TDI = np.asscalar(quest.TDI.values)

            '''
            Model-Based Personalization Scheme of an Artificial Pancreas for Type 1 Diabetes Applications
            Joon Bok Lee, Eyal Dassau, Senior Member, IEEE, Dale E. Seborg, Member, IEEE, and Francis J. Doyle III*, Fellow, IEEE
            2013 American Control Conference (ACC) Washington, DC, USA, June 17-19, 2013
            '''
            s_fb = 0.5 * self.patient_TDI / 24  # (6)
            c = .0664                           # (5)
            '''
            Guidelines for Optimal Bolus Calculator Settings in Adults
            John Walsh, P.A., Ruth Roberts, M.A.,2 and Timothy Bailey, M.D., FACE, C.P.I.1
            J Diabetes Sci Technol. 2011 Jan; 5(1): 129–135. 
            '''
            k_i = 1960 / self.patient_TDI       # (3),(4)
            K = k_i * c * s_fb                  # (2)
            tau_1 = 247                         # (13)
            tau_2 = 210                         # (14)
            theta = 93.5                        # (12)

            self.k_c = 2 * self.patient_basal * 298/((self.tau_c + 93.5)*1960*.5)            # (22) --> Proportional Gain
            print("k_c: {}".format(self.k_c))
            self.tau_i = 458               # (20) --> Integral Gain
            self.tau_d = 113               # (21) --> Derivative Gain

        else:
            raise LookupError("Invalid patient name.")

        self.ierror = 0

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        action = self._policy(
            pname,
            observation.CGM,
            self.prev1, 
            sample_time)

        # for the derivative
        self.prev1 = observation.CGM
        return action


    def _policy(self, pname, glucose, prev1, env_sample_time):
        error = np.asscalar((glucose - self.target))    # error
        self.ierror += error                            # integral error
        deriv = (glucose - prev1) / env_sample_time     # derivative

        pterm = self.k_c * error
        iterm = self.k_c / self.tau_i * self.ierror
        dterm = self.k_c * self.tau_d * deriv

        bolus = pterm + iterm + dterm
        basal = self.patient_basal
        if bolus + basal < 0:
            bolus = -1 * basal
        return Action(basal=basal, bolus=bolus)

class PIDController(Controller):
    '''
    Naive PID controller with gains set directly.

    Parameters
    ----------
    controller_params: tuple of float
        (kp, ki, kd, target)
    '''
    def __init__(self, controller_params, name):
        self.prev_glucose = 140
        self.kp = controller_params[0]
        self.ki = controller_params[1]
        self.ierror = 0
        self.kd = controller_params[2]
        self.target = controller_params[3]
        self.low = controller_params[4]
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.quest = pd.read_csv(CONTROL_QUEST)
        if any(self.patient_params.Name.str.match(name)):
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            self.patient_basal = np.asscalar(params.u2ss.values) * np.asscalar(params.BW.values) / 6000
        else:
            raise KeyError("No patient with that ID exists!")

    def policy(self, observation, reward, done, **kwargs):
        self.state = observation

        # find and integrate error
        error = np.asscalar((observation.CGM - self.target))
        self.ierror += error

        # calculate derivative
        if 'sample_time' in kwargs:
            sample_time = kwargs.get('sample_time', 1)
        else:
            raise KeyError("sample_time not in arguments")
        deriv = (observation.CGM - self.prev_glucose) / sample_time

        # suspension if low
        if observation.CGM <= self.low:
            basal = 0
            bolus = 0
        else:
            basal = self.patient_basal
            bolus = self.kp * error + self.ki * self.ierror + self.kd * deriv

        action = Action(basal=basal, bolus=bolus)
        return action

    def reset(self):
        self.state = self.init_state