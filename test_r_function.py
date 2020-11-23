import gym
import random
import torch
import numpy as np
from dqn_agent import DQNAgent
from replay_buffer2 import ReplayBuffer
from agent_iql import Agent, mkdir
import argparse
import json








def main(args):
    with open (args.param, "r") as f:
        config = json.load(f)
    agent_r = Agent(state_size=8, action_size=4,  config=config)
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = DQNAgent(state_size=8, action_size=4, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    agent_r.load("models/35000-")
    n_episodes = 40
    max_t = 500
    eps = 0
    # eps = 1   # random policy
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            
            action = agent.act(state, eps)
            next_state, _, done, _ = env.step(action)
            state = torch.from_numpy(state).float().unsqueeze(0).to(agent_r.device)
            action = torch.from_numpy(np.array(action)).float().unsqueeze(0).to(agent_r.device)
            action = action.type(torch.int64)
            reward = agent_r.R_local(state).gather(1, action.unsqueeze(0))
            score += reward.item()
            state = next_state
            # env.render()
            if done:
                print("Episode {}  Reward {}".format(i_episode, score))
                break




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="hypersearch", type=str)
    parser.add_argument('--lr_iql_q', default=1e-5, type=float)
    parser.add_argument('--lr_iql_r', default=1e-5, type=float)
    parser.add_argument('--lr_q_sh', default=1e-5, type=float)
    parser.add_argument('--lr_pre', default=5e-4, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--fc1_units', default=256, type=int)
    parser.add_argument('--fc2_units', default=256, type=int)
    parser.add_argument('--fc3_units', default=256, type=int)
    parser.add_argument('--clip', default=-1, type=int)
    parser.add_argument('--mode', default="iql", type=str)
    arg = parser.parse_args()
    mkdir("", arg.locexp)
    main(arg)

