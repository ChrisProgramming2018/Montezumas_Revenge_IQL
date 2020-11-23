import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent_iql import Agent, mkdir
import json
import argparse
from dqn_agent import DQNAgent
from helper import FrameStack
from replay_buffer import ReplayBuffer




def dqn(args, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
	n_episodes (int): maximum number of training episodes
	max_t (int): maximum number of timesteps per episode
	eps_start (float): starting value of epsilon, for epsilon-greedy action selection
	eps_end (float): minimum value of epsilon
	eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    with open ("param.json", "r") as f:
        config = json.load(f)
    agent_r = Agent(state_size=200, action_size=7,  config=config)
    agent = DQNAgent(state_size=200, action_size=7,  config=config)
    agent_r.load("models-23_11_2020_12:59:20/8000-")
    memory = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
    env = gym.make('MontezumaRevenge-v0')
    env.seed(0)
    env  = FrameStack(env, args)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        obses = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(obses, eps)
            next_obses, reward, done, _ = env.step(action)
            memory.add(obses, action, reward, next_obses, done, done)
            agent.step(memory)
            action = torch.from_numpy(np.array(action)).float().unsqueeze(0).to(agent_r.device)
            action = action.type(torch.int64)
            obses = torch.from_numpy(obses).float().unsqueeze(0).to(agent_r.device)
            state =  agent_r.encoder.create_vector(obses)
            reward = agent_r.R_local(state).gather(1, action.unsqueeze(0))
            state = next_state
            score += reward.item()
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="kuka_block_grasping-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--buffer_size', default=20000, type=int)
    args = parser.parse_args()
    scores = dqn(args)
