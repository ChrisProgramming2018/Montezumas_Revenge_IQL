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
    memory = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
    memory.load_memory(config["buffer_path"])
    print("memory idx ",memory.idx)  
    agent = Agent(state_size=200, action_size=7,  config=config) 
    # agent.load("models-27_11_2020_13:35:01/21000-")
    memory_t = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
    memory_t.load_memory(config["expert_buffer_path"])
    #memory.idx = config["idx"] 
    #memory_t.idx = config["idx"] * 7
    if config["mode"] == "pretrain":
        print("Pretrain Encoder and Predicter")
        for t in range(config["predicter_time_steps"]):
            text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
            print(text, end = '')
            agent.pretrain(memory)
            if t % int(config["eval"]) == 0:
                agent.save("models-{}/{}-".format(dt_string, t))
                agent.test_predicter(memory)

    print("memory idx ",memory.idx)  
    print("memory idx ",memory_t.idx)  
    #print("memory_expert idx ",memory_t.idx)
    a = 0
    for idx in range(15):
        print(memory_t.actions[idx])
        
    for t in range(config["predicter_time_steps"]):
        text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
        print(text, end = '')
        agent.learn(memory, memory_t)
        if t % int(config["eval"]) == 0:
            agent.save("models-{}/{}-".format(dt_string, t))
            agent.test_predicter(memory)
            agent.test_q_value(memory)
