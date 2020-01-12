from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action
from datetime import timedelta
import multiprocess

class MyController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state
    def policy(self, observation, reward, done, **info):
        self.state = observation
        action = Action(basal=0, bolus=0)
        return action
    def reset(self):
        self.state = self.init_state

myctrl = MyController(0)
time =  timedelta(hours=3)

simulate(
    sim_time = time,
    controller=myctrl,
    parallel=True,
    )