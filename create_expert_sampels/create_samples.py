import gym
import time
import sys
from helper import FrameStack
from replay_buffer import ReplayBuffer
import argparse
import readchar


def map(action):
    """ Maps the the given string to the corresponding
        action for the env (0-6)
        Args:
            param1(str): action
        Return:
            action in int
    """
    if action == "w":
        return 0
    if action == "s":
        return 1
    if action == "d":
        return 2
    if action == "a":
        return 3
    if action == "f":
        return 4
    if action == " ":
        return 5
    if action == "x":
        return 6


def main(args):
    """ Creates expert examples for MontezumaRevenge env for Inverse Q Learning
    """
    print(sys.version)
    env = gym.make(args.env_name)
    print(env.observation_space)
    memory = ReplayBuffer((3, 84, 84), (1, ), args.buffer_size, args.device)
    if args.continue_samples:
        memory.load_memory(args.path)
        print("continue with {}  samples".format(memory.idx))
    env = FrameStack(env, args)
    state = env.reset()
    done = False
    steps = 0
    score = 0
    while True:
        steps += 1
        while True:
            try:
                input_action = readchar.readchar()
                if input_action == "p":
                    print("wanna save buffer press s")
                    io = readchar.readchar()
                    if io == "s":
                        memory.save_memory("expert_policy-{}/".format(steps))
                    print("exit")
                    env.close()
                    return
                action = map(input_action)
                if action > 13:
                    continue
                break
            except Exception:
                continue

        env.render()
        next_state, reward, done, _ = env.step(action)
        memory.add(state, action, state)
        state = next_state
        print("action", action)
        score += reward
        if steps % 100 == 0:
            memory.save_memory("expert_policy-{}/".format(steps))
        if done:
            print("Episodes exit with score {}".format(score))
            break
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="MontezumaRevenge-v0", type=str)
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--buffer_size', default=20000, type=int)
    parser.add_argument('--continue_samples', default=False, type=bool)
    parser.add_argument('--path', default='expert_policy/', type=str)
    arg = parser.parse_args()
    main(arg)
