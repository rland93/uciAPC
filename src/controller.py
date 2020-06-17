from simglucose.controller.base import Controller, Action
import pkg_resources
import numpy as np
import pandas as pd
import logging
import math
import cvxpy as cp
import logging
logging.basicConfig(filename='MPC_Controller.log',level=logging.DEBUG,format='%(asctime)s %(message)s')

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

class MPCNaive(Controller):
    '''
    control params: tuple, (targetBG, lowBG)
    '''
    def __init__(self, controller_params, name):
        self.targetBG = controller_params[0]
        self.lowBG = controller_params[1]
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.quest = pd.read_csv(CONTROL_QUEST)
        if any(self.patient_params.Name.str.match(name)):
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            # Patient Basal
            self.patient_basal = np.asscalar(params.u2ss.values) * np.asscalar(params.BW.values) / 6000
        else:
            raise KeyError("No patient with that ID exists! (pt params)")
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            # Total Daily Insulin
            self.TDI = np.asscalar(quest.TDI.values)
        else:
            raise KeyError("No patient with that ID exists! (pt quest)")
        
        # predict horizon
        self.T = controller_params[2]
        # control horizon
        self.M = 10
        # Van Heusden's Control Relevant Model
        self.p1 = 0.98
        self.p2 = 0.965
        self.g = -90 * (1-self.p1)*(1-self.p2)*(1-self.p2)
        # A matrix used to calculate state from previous states
        self.A = np.array([   
                [self.p1+2*self.p2, -2*self.p1*self.p2-self.p2*self.p2, self.p1*self.p2*self.p2],
                [1,0,0],
                [0,1,0]])
        # B matrix used to calculate effect of control inputs on state
        self.B = 1800 * self.g / self.TDI * np.array([[1],[0],[0]])
        # "known" state is 3 vals behind current state
        self.C = np.array([0,0,1])
        self.state = 0
        self.prev_doses = []

    def policy(self, observation, reward, done, **kwargs):
        ''' define vars and solve optimization problem '''
        self.state = np.asscalar(observation.CGM)

        '''list of previous doses'''
        self.prev_doses.append()



        # state var
        x = cp.Variable((3, self.T+1))
        u = cp.Variable((1, self.T))
        # init cost and constraints
        cost = 0
        constraints = []
        # build costs, constraints across horizon
        for t in range(self.T):
            # quadratic cost away from target
            cost += cp.sum_squares(x[:,t] - self.targetBG) + cp.sum_squares(u[:,t])
            constraints += [
                x[:,t+1] == self.A @ x[:,t] + self.B @ u[:,t], # state dependence
                u[:,t] >= 0,    # dose is non-negative
                u[:,t] <= 1.0,  # single dose cannot be larger than 1u
                u[:,self.M:self.T] == 0 # no control action beyond control horizon
            ]
        # we add the constraint that we are starting from the observation.
        constraints += [
            x[:,0] == self.state
        ]
        # solve problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver='ECOS')

        '''logging'''
        logging.debug("Start BG: {}".format(self.state))
        logging.debug("Problem Status: {}".format(problem.status))
        logging.debug("\tSetup Time: {}".format(problem.solver_stats.setup_time))
        logging.debug("\tSolved in: {}".format(problem.solver_stats.solve_time))
        logging.debug("\tNumber of iterations: {}".format(problem.solver_stats.num_iters))

        if self.state >= self.lowBG:
            # take only first control action
            bolus = u.value[0,1]
            basal = self.patient_basal
        else:
            bolus = 0
            basal = 0
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