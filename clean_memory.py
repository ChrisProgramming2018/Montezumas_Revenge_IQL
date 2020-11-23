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



def clean(config):
    """

    """
    memory = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
    memory_t = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
    memory.load_memory(config["buffer_path"])
    #memory_t.load_memory(config["expert_buffer_path"])
    memory.idx = config["idx"] 
    #memory_t.idx = config["idx"] * 4
    print("memory idx ",memory.idx)  
    #print("memory_expert idx ",memory_t.idx)
    a = 0
    for idx in range(100):
        if memory.actions[idx] < 6:
            memory_t.add(memory.obses[idx], memory.actions[idx], 0, memory.next_obses[idx], 1, 1)

    print(memory_t.idx)
    memory_t.save_memory("cleaned")
    sys.exit()
        
    for t in range(config["predicter_time_steps"]):
        text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
        print(text, end = '')
        agent.learn(memory, memory_t)
        #if t % 1 == 0:
        #print(text)
        if t % 500 == 0:
            agent.save("models-{}/{}-".format(dt_string, t))
            agent.test_predicter(memory)
            # agent.test_q_value(memory)
