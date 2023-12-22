"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt


from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

import gym
import pickle

DEFAULT_NUM_DRONES = 3
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'data'
DEFAULT_COLAB = False
RENDER=False

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

def main(ARGS):

    env = gym.make(f'{ARGS.env}-aviary-v0',
                        aggregate_phy_steps=1,
                        obs=DEFAULT_OBS,
                        act=DEFAULT_ACT,
                        gui=True
                        )
    env.seed(42)
    env.action_space.seed(42)
    env.observation_space.seed(42)
    np.random.seed(42)
    random.seed(42)


    trajs = []
    for i in range(5000):
        total_reward = 0
        env.reset()
        print('collecting '+str(i)+' th trail...')
        obss = []
        acts = []
        rews = []
        dones = []
        # import pdb;pdb.set_trace()
        while True:

            state = env.getDroneStateVector(0)
            obss.append(state)
            wp_idx = min(int(env.step_counter / env.AGGR_PHY_STEPS), env.TRAJ_STEPS - 1)
            action, _, _ = env.ctrl.computeControl(control_timestep=env.TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=env.TARGET_POSITION[wp_idx, :]
                                                 )
            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action+0.1*np.random.randn(4,))

            acts.append(action)
            rews.append(reward)
            total_reward += reward


            #### Printout ##############################################
            if RENDER:
                env.render()
            if done:
                break
        # import pdb;pdb.set_trace()
        obss = np.array(obss)
        acts = np.array(acts)
        rews = np.array(rews)
        dones = np.array(dones)
        # import pdb;pdb.set_trace()
        # plt.plot(obss[:,0],obss[:,1])
        # plt.show()
        traj = {'observations': obss, 'actions': acts, 'rewards': rews}
        trajs.append(traj)

        print(total_reward)
    import pdb;pdb.set_trace()
    #### Close the environment #################################
    env.close()
    # with open(DEFAULT_OUTPUT_FOLDER+ARGS.env+'.pkl', "wb") as f:
    #     pickle.dump(trajs, f)

    for num in [50,100,200,300,400,500,600,700,800,900,1000,1200,1500,2000,2500,3000,3500,4000,4500,5000]:
        with open(os.path.join(DEFAULT_OUTPUT_FOLDER,ARGS.env+'_'+str(num)+'.pkl'), "wb") as f:
            pickle.dump(trajs[:num], f)
    print('env '+ARGS.env+' collecting done!')
if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    # parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    # parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    # parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    # parser.add_argument('--aggregate',          default=DEFAULT_AGGREGATE,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    # parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    # parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    # parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    # parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    # parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--env',                           type=str,            help='The running env', metavar='')

    ARGS = parser.parse_args()

    main(ARGS)