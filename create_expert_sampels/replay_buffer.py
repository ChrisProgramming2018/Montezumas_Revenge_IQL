# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mkdir


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.uint8)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.k = 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action,  next_obs, done):
        self.k += 1
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        actions = self.actions[idxs]
        return obses, actions, next_obses

    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """
        # check directory exist and create it
        mkdir("", filename)

        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)

        with open(filename + '/actions.npy', 'wb') as f:
            np.save(f, self.actions)

        with open(filename + '/next_obses.npy', 'wb') as f:
            np.save(f, self.next_obses)

        with open(filename + '/index.txt', 'w') as f:
            f.write("{}".format(self.idx))

        print("save buffer to {}".format(filename))

    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """
        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)

        with open(filename + '/actions.npy', 'rb') as f:
            self.actions = np.load(f)

        with open(filename + '/next_obses.npy', 'rb') as f:
            self.next_obses = np.load(f)

        with open(filename + '/index.txt', 'r') as f:
            self.idx = int(f.read())
