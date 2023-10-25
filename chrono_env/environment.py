import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import math
import time
import numpy as np
import json
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
import time
import yaml
from argparse import Namespace
from utils import *
# from regulators.pure_pursuit import *
# from regulators.path_follow_mpc import *
# from models.extended_kinematic import ExtendedKinematicModel
# from models.GP_model_single import GPSingleModel
# from models.configs import *
# from helpers.closest_point import *
# from helpers.track import Track


class ChronoEnv:
    def __init__(self) -> None:
        self.step_size = 2e-3

    def make(self):
        self.my_hmmwv = init_vehicle(self)