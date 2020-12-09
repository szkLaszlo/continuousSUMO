"""
Usage:
1. put this file to the directory of the gym project. It will add the sumo path to the environments, thus enabling
the usage of the project as gym environment.
2. update the project entry point below
"""
import os
import sys

from gym.envs.registration import register

register(
    id='SUMOEnvironment-v0',
    entry_point='continuousSUMO.sumoGym.environment:SUMOEnvironment',
)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    os.system('export SUMO_HOME="/usr/share/sumo/tools"')
