import sys
from replay_buffer import ReplayBuffer
from iql_agent import Agent




def train(env, config):
    """
    """
    # memory = ReplayBuffer((3,80,80), (1,), config["expert_buffer_size"], config["device"])
    memory = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["device"])
    memory.load_memory(config["buffer_path"])
    agent = Agent(200, 1, config["action_dim"], config)
    memory.idx = 1000
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
    

    
    for i_episode in range(config['episodes']):
        text = "Inverse Episode {}  \ {} \r".format(i_episode, config["episodes"])
        print(text, end = '')
        agent.learn(memory)
        if i_episode % 50 == 0:
            print(text)
            agent.eval_policy(env)
            agent.test_q_value(memory)
            agent.test_predicter(memory)
            agent.save("pytorch_models/")
