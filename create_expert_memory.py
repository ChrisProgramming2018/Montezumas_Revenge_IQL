from replay_buffer import ReplayBuffer
import json



with open ("param.json", "r") as f:
    param = json.load(f)

config = param

memory = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
memory.load_memory(config["buffer_path"])
memory_expert = ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], config["image_pad"], config["device"])
print(memory_expert.idx)

for idx in range(memory.idx):
    for a in range(7):
        memory_expert.add(memory.obses[idx], a, memory.rewards[idx], memory.next_obses[idx], memory.not_dones[idx], memory.not_dones_no_max[idx])



print(memory_expert.idx)
memory_expert.save_memory("all_action_expert")
