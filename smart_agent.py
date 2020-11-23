import gym
import random
import torch
import json
import argparse
import numpy as np
from dqn_agent import DQNAgent
from iql_agent import mkdir






def main(args):
    with open (args.param, "r") as f:
        config = json.load(f)
    
    env = gym.make('MontezumaRevenge-v0')
    env.seed(0)
    
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = DQNAgent(state_size=200, action_size=7, config=config)
    # agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    n_episodes = 40
    max_t = 500
    eps = 0
    action_size = 7
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            #action = agent.act(state, eps)
            action = random.choice(np.arange(action_size))
            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
            env.render()
            if done:
                break
        print("Episode {}  Reward {} Steps {}".format(i_episode, score, t))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    arg = parser.parse_args()
    main(arg)
