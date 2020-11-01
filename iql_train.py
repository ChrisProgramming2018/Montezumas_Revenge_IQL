import sys
import time 
from replay_buffer import ReplayBuffer
from iql_agent import Agent


def time_format(sec):
    """

    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)




def train(env, config):
    """
    """
    # memory = ReplayBuffer((3,80,80), (1,), config["expert_buffer_size"], config["device"])
    memory = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
    memory.load_memory(config["buffer_path"])
    agent = Agent(200, 1, config["action_dim"], config)
    memory.idx = 900
    print("size buffer ", memory.idx)
    if config["train_predicter"]:
        for t in range(config["predicter_time_steps"]):
            text = "Train Predicter {}  \ {} \r".format(t, config["predicter_time_steps"])
            print(text, end = '')
            agent.learn_predicter(memory)
            if t % 500 == 0:
                agent.test_predicter(memory)

        agent.save("pytorch_models/")
        return
    else:
        # agent.load("pytorch_models/")
        print("continue load predicter model")


    t0 = time.time()
    for i_episode in range(config['episodes']):
        text = "Inverse Episode {}  \ {}  Time: {}  \r".format(i_episode, config["episodes"],time_format(time.time()-t0))
        print(text)
        agent.learn(memory)
        if i_episode % 250 == 0:
            print(text)
            agent.eval_policy(env)
            agent.test_q_value(memory)
            agent.test_predicter(memory)
            agent.save("pytorch_models/")
