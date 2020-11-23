from replay_buffer import ReplayBuffer
from agent_iql import Agent
import sys
from datetime import datetime
import time


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
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    t0 = time.time()
    #memory = ReplayBuffer((8,), (1,), config["expert_buffer_size"], config["device"])
    memory = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
    memory.load_memory(config["buffer_path"])
    agent = Agent(state_size=200, action_size=7,  config=config) 
    #agent.load("models/42000-")
    memory_t = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
    memory_t.load_memory(config["expert_buffer_path"])
    memory.idx = config["idx"] 
    memory_t.idx = config["idx"] * 4
    print("memory idx ",memory.idx)  
    #print("memory_expert idx ",memory_t.idx)
    a = 0
    for idx in range(100):
        print(memory.actions[idx])
        if memory.actions[idx] > a:
            a = memory.actions[idx]

        
    for t in range(config["predicter_time_steps"]):
        text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
        print(text, end = '')
        agent.learn(memory, memory_t)
        #if t % 1 == 0:
        #print(text)
        if t % 500 == 0:
            agent.save("models-{}/{}-".format(dt_string, t))
            agent.test_predicter(memory)
            agent.test_q_value(memory)
