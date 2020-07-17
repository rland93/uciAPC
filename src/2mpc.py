
import cvxpy as cp
import numpy as np
import pandas as pd
import pathos.pools
import time
import copy
import pkg_resources

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from simglucose.controller.base import Controller, Action
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.sim_engine import SimObj
from simglucose.sensor.cgm import CGMSensor, CGMNoise
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.env import T1DSimEnv
from datetime import timedelta, datetime
import analysis

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')
CONTROL_QUEST = pkg_resources.resource_filename(
    'simglucose', 'params/Quest.csv')

class MPC_filter(Controller):
    def __init__(self, patientname, target, low, horizon):
        self.target = target
        self.low = low
        self.horizon = horizon
        self.params = pd.read_csv(PATIENT_PARA_FILE)
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.action_count = 0

        # model parameters
        if any(self.params.Name.str.match(patientname)):
            self.basal = self.params[self.params.Name.str.match(patientname)].BW.values / 6000
        else:
            raise LookupError("Invalid patient name. No patient by that name found in PARA file.")
        if any(self.quest.Name.str.match(patientname)):
            self.TDI = self.quest[self.quest.Name.str.match(patientname)].TDI.values
        else:
            raise LookupError("Invalid patient name. No patient by that name found in QUEST file.")

        self.p1 = 0.98
        self.p2 = 0.965
        self.g = -90 * (1-self.p1)*(1-self.p2)*(1-self.p2)
        # model
        self.A = np.array([   
                [self.p1+2*self.p2, -2*self.p1*self.p2-self.p2*self.p2, self.p1*self.p2*self.p2],
                [1,0,0],
                [0,1,0]])
        # actuation
        self.B = 1800 * self.g / self.TDI * np.array([[1],[0],[0]])
        # observation
        self.C = np.array([[0],[0],[1]])
        # action
        self.u = np.array([0,0,0])


        # process noise covariance
        self.Q = np.eye(3)
        self.R = np.eye(3)
        # error covariance R3x3
        self.covariance = np.eye(3)
        self.proj_covariance = np.eye(3)
        # state, state estimator R1x3
        self.state = np.array([0,0,0])
        self.proj_state = np.array([0,0,0])
        # kalman gain R3x3
        self.K = np.eye(3)
        # cgm 3-tuple
        self.cgm = np.array([0,0,0])

    def policy(self, observation, reward, done, **info):
        self.cgm[2] = self.cgm[1]
        self.cgm[1] = self.cgm[0]
        self.cgm[0] = observation.CGM

        if self.action_count > 4:
            print("Action {}".format(str(self.action_count).rjust(5)))

            # KF Predict
            self.proj_state = self.A @ self.state + self.B @ self.u
            self.proj_covariance = self.A @ self.covariance @ self.A.transpose() + self.Q

            # KF Correct
            print("proj_covariance: {}".format(self.proj_covariance.shape))
            print("C: {}".format(self.C.shape))
            print("Ct: {}\n\n\n".format(self.C.transpose().shape))

            
            a = np.array([[1],[1],[1]]) 
            print("test {}".format(np.matmul(a, np.eye(3))))



            print("R: {}".format(self.R.shape))
            print("proj_covariance: {}".format(self.proj_covariance.shape))
            print("inv: {}".format(np.invert(self.C @ self.proj_covariance @ self.C.transpose())))


            self.K = self.proj_covariance @ self.C.transpose() @ np.invert(self.C @ self.proj_covariance @ self.C.transpose() + self.R)
            self.state = self.proj_state + self.K @ (self.cgm - self.C @ self.proj_state)
            self.covariance = (np.eye(3) - self.K @ self.C) @ self.proj_covariance


            print("\tCGM={}".format(self.cgm))
            print("\tState={}".format(self.state))

            s = cp.Variable((3, self.horizon+1))
            u = cp.Variable((1, self.horizon))
            cost = 0
            constr = []
            for t in range(self.horizon):
                cost += cp.sum_squares(s[:,t] - self.target) + cp.sum_squares(u[:,t])
                constr += [
                    s[:,t + 1] == self.A @ s[:,t] + self.B @ u[:,t],
                    u[:,t] >= 0,
                ]
            constr += [s[:,0] == self.state]
            problem = cp.Problem(cp.Minimize(cost), constr)
            problem.solve(verbose=False, solver='SCS')
            status = problem.status
            print("\tStatus: {}".format(status))
            print("\tOptimal bolus action is {}".format(u.value[0,0]))
            action = Action(basal=self.basal, bolus=u.value[0,0])

        else:
            # initialize state
            self.state = self.cgm
            # initialize error cov
            self.covariance = np.eye(3)
            # initial bolus is 0
            self.u = np.array([0])
        
        action = Action(bolus=self.u[0], basal=self.basal)
        self.action_count += 1
        return action
    def reset(self):
        pass

'''
        if self.action_count > 4:
            
            s = cp.Variable((3, self.horizon+1))
            u = cp.Variable((1, self.horizon))
            cost = 0
            constr = []
            for t in range(self.horizon):
                cost += cp.sum_squares(s[:,t] - self.target) + cp.sum_squares(u[:,t])
                constr += [
                    s[:,t + 1] == self.A @ s[:,t] + self.B @ u[:,t],
                    u[:,t] >= 0,
                ]
            constr += [s[:,0] == self.cgm]
            problem = cp.Problem(cp.Minimize(cost), constr)
            problem.solve(verbose=False, solver='SCS')
            status = problem.status
            print("\tStatus: {}".format(status))
            action = Action(basal=self.basal, bolus=0)
            print("\tOptimal bolus action is {}".format(u.value[0,0]))
            action = Action(basal=self.basal, bolus=u.value[0,0])
            # store bolus
            self.bolus = u.value[0,0]
            # store state
            self.state = s.value[:,0]
            
            fig = plt.figure()
            ax = fig.add_subplot(211)
            plt.plot(s[0,:].value, label='state')
            plt.subplot(2,1,2)
            plt.plot(u[0,:].value, label='control')
            plt.show()
            

        else:
            action = Action(basal=self.basal, bolus=0)
'''

class MPC_filter(Controller):
    def __init__(self, patientname, target, low, horizon):
        self.target = target
        self.low = low
        self.horizon = horizon
        self.params = pd.read_csv(PATIENT_PARA_FILE)
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.action_count = 0

        # model parameters
        if any(self.params.Name.str.match(patientname)):
            self.basal = self.params[self.params.Name.str.match(patientname)].BW.values / 6000
        else:
            raise LookupError("Invalid patient name. No patient by that name found in PARA file.")
        if any(self.quest.Name.str.match(patientname)):
            self.TDI = self.quest[self.quest.Name.str.match(patientname)].TDI.values
        else:
            raise LookupError("Invalid patient name. No patient by that name found in QUEST file.")

        self.p1 = 0.98
        self.p2 = 0.965
        self.g = -90 * (1-self.p1)*(1-self.p2)*(1-self.p2)
        # model
        self.A = np.array([   
                [self.p1+2*self.p2, -2*self.p1*self.p2-self.p2*self.p2, self.p1*self.p2*self.p2],
                [1,0,0],
                [0,1,0]])
        # actuation
        self.B = 1800 * self.g / self.TDI * np.array([[1],[0],[0]])
        # observation
        self.C = np.array([[0],[0],[1]])
        # action
        self.u = np.array([0,0,0])


        # process noise covariance
        self.Q = np.eye(3)
        self.R = np.eye(3)
        # error covariance R3x3
        self.covariance = np.eye(3)
        self.proj_covariance = np.eye(3)
        # state, state estimator R1x3
        self.state = np.array([0,0,0])
        self.proj_state = np.array([0,0,0])
        # kalman gain R3x3
        self.K = np.eye(3)
        # cgm 3-tuple
        self.cgm = np.array([0,0,0])

    def policy(self, observation, reward, done, **info):
        self.cgm[2] = self.cgm[1]
        self.cgm[1] = self.cgm[0]
        self.cgm[0] = observation.CGM

        if self.action_count > 4:
            print("Action {}".format(str(self.action_count).rjust(5)))

            # KF Predict
            self.proj_state = self.A @ self.state + self.B @ self.u
            self.proj_covariance = self.A @ self.covariance @ self.A.transpose() + self.Q

            # KF Correct
            print("proj_covariance: {}".format(self.proj_covariance.shape))
            print("C: {}".format(self.C.shape))
            print("Ct: {}\n\n\n".format(self.C.transpose().shape))

            
            a = np.array([[1],[1],[1]]) 
            print("test {}".format(np.matmul(a, np.eye(3))))



            print("R: {}".format(self.R.shape))
            print("proj_covariance: {}".format(self.proj_covariance.shape))
            print("inv: {}".format(np.invert(self.C @ self.proj_covariance @ self.C.transpose())))


            self.K = self.proj_covariance @ self.C.transpose() @ np.invert(self.C @ self.proj_covariance @ self.C.transpose() + self.R)
            self.state = self.proj_state + self.K @ (self.cgm - self.C @ self.proj_state)
            self.covariance = (np.eye(3) - self.K @ self.C) @ self.proj_covariance


            print("\tCGM={}".format(self.cgm))
            print("\tState={}".format(self.state))

            s = cp.Variable((3, self.horizon+1))
            u = cp.Variable((1, self.horizon))
            cost = 0
            constr = []
            for t in range(self.horizon):
                cost += cp.sum_squares(s[:,t] - self.target) + cp.sum_squares(u[:,t])
                constr += [
                    s[:,t + 1] == self.A @ s[:,t] + self.B @ u[:,t],
                    u[:,t] >= 0,
                ]
            constr += [s[:,0] == self.state]
            problem = cp.Problem(cp.Minimize(cost), constr)
            problem.solve(verbose=False, solver='SCS')
            status = problem.status
            print("\tStatus: {}".format(status))
            print("\tOptimal bolus action is {}".format(u.value[0,0]))
            action = Action(basal=self.basal, bolus=u.value[0,0])

        else:
            # initialize state
            self.state = self.cgm
            # initialize error cov
            self.covariance = np.eye(3)
            # initial bolus is 0
            self.u = np.array([0])
        
        action = Action(bolus=self.u[0], basal=self.basal)
        self.action_count += 1
        return action
    def reset(self):
        pass

class MPC(Controller):
    def __init__(self, patientname, target, low, horizon):
        self.target = target
        self.low = low
        self.horizon = horizon
        self.params = pd.read_csv(PATIENT_PARA_FILE)
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.action_count = 0

        # patient lookups
        if any(self.params.Name.str.match(patientname)):
            self.basal = self.params[self.params.Name.str.match(patientname)].BW.values / 6000
        else:
            raise LookupError("Invalid patient name. No patient by that name found in PARA file.")
        if any(self.quest.Name.str.match(patientname)):
            self.TDI = self.quest[self.quest.Name.str.match(patientname)].TDI.values
        else:
            raise LookupError("Invalid patient name. No patient by that name found in QUEST file.")

        # scalar constants
        self.p1 = 0.98
        self.p2 = 0.965
        self.g = -90 * (1-self.p1)*(1-self.p2)*(1-self.p2)

        # Open loop behavior R3x3
        self.A = np.array([   
                [self.p1+2*self.p2, -2*self.p1*self.p2-self.p2*self.p2, self.p1*self.p2*self.p2],
                [1,0,0],
                [0,1,0]])
        # Actuator effect
        self.B = 1800 * self.g / self.TDI * np.array([[1],[0],[0]])
        # CGM stored here
        self.cgm = np.array([0,0,0])

    def policy(self, observation, reward, done, plot=True, **info):
        self.cgm[2] = self.cgm[1]
        self.cgm[1] = self.cgm[0]
        self.cgm[0] = observation.CGM
        bolus_multiplier = info['sample_time'] / 60

        if self.action_count > 4:
            print("\tAction {}".format(str(self.action_count).rjust(5)))
            s = cp.Variable((3, self.horizon+1))
            u = cp.Variable((1, self.horizon))
            cost = 0
            constr = []
            constr += [s[:,0] == self.cgm]
            for t in range(self.horizon):
                cost += cp.sum_squares(s[:,t] - self.target) + cp.sum_squares(u[:,t])
                constr += [
                    s[:,t + 1] == self.A @ s[:,t] + self.B @ u[:,t] * bolus_multiplier,
                    u[:,t] >= 0,
                ]
            
            problem = cp.Problem(cp.Minimize(cost), constr)
            problem.solve(verbose=False, solver='SCS')
            print("\tStatus: {}".format(problem.status))
            self.bolus = u.value[0,0] * bolus_multiplier
            print("\tOptimal bolus action is {}".format(self.bolus))
            action = Action(basal=self.basal, bolus=self.bolus)
            
            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(211)
                plt.plot(range(s[0,:].size), s[0,:].value, label='state')
                plt.subplot(2,1,2)
                plt.step(range(u[0,:].size), u[0,:].value * bolus_multiplier, label='control')
                plt.show()
            
        else:
            action = Action(basal=self.basal, bolus=0)
        self.action_count += 1
        return action

        def reset(self):
            pass




def sim(sim_object):
    print('Simulating...')
    sim_object.simulate()
    return sim_object.results()

if __name__ == '__main__':
    start_time = datetime(2020, 1,1,0,0,0)

    # sim env
    pname = 'adult#001'
    patient = T1DPatient.withName(pname)
    sensor = CGMSensor.withName('Dexcom', seed=92612)
    pump = InsulinPump.withName('Insulet')
    meals = [(1,45)]
    scenario = CustomScenario(start_time=start_time, scenario=meals)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    controller = MPC(pname, 120, 80, 10)
    s1 = SimObj(env, controller, timedelta(hours=6), animate=False)
    p_start = time.time()
    results = sim(s1)
    print('Simulation took {} seconds.'.format(time.time() - p_start))

    fig = analysis.single_pt_ts(results)
    plt.show()