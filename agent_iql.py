import os
import sys
import time
import numpy as np
import random
import gym
import gym.wrappers
from collections import namedtuple, deque
from models import QNetwork, Classifier, RNetwork, DQNetwork, Encoder
import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
torch.set_printoptions(threshold=5000)
import logging
from datetime import datetime
from utils import mkdir, write_into_file 



now = datetime.now()    
dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
mkdir("","search_results")
logging.basicConfig(filename="search_results/{}.log".format(dt_string), level=logging.DEBUG)


class Agent():
    def __init__(self, state_size, action_size, config):
        self.env_name = config["env_name"]
        self.state_size = state_size
        self.action_size = action_size
        self.seed = config["seed"]
        self.clip = config["clip"]
        self.device = 'cuda'
        print("Clip ", self.clip)
        print("cuda ", torch.cuda.is_available())
        self.double_dqn = config["DDQN"]
        print("Use double dqn", self.double_dqn)
        self.lr_pre = config["lr_pre"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.tau = config["tau"]
        print("self tau", self.tau)
        self.gamma = 0.99
        self.fc1 = config["fc1_units"]
        self.fc2 = config["fc2_units"]
        self.fc3 = config["fc3_units"]
        self.qnetwork_local = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3, self.seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size,self.fc1, self.fc2, self.fc3,  self.seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1)
        
        self.q_shift_local = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3, self.seed).to(self.device)
        self.q_shift_target = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3, self.seed).to(self.device)
        self.optimizer_shift = optim.Adam(self.q_shift_local.parameters(), lr=self.lr)
        self.soft_update(self.q_shift_local, self.q_shift_target, 1)
         
        self.R_local = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3,  self.seed).to(self.device)
        self.R_target = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3, self.seed).to(self.device)
        self.optimizer_r = optim.Adam(self.R_local.parameters(), lr=self.lr)
        self.soft_update(self.R_local, self.R_target, 1) 

        self.expert_q = DQNetwork(state_size, action_size, seed=self.seed).to(self.device)
        #self.expert_q.load_state_dict(torch.load('checkpoint.pth'))
        self.steps = 0
        self.predicter = QNetwork(state_size, action_size, self.fc1, self.fc2, self.fc3, self.seed).to(self.device)
        self.optimizer_pre = optim.Adam(self.predicter.parameters(), lr=self.lr_pre)
        self.encoder = Encoder(config).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), self.lr)
        pathname = "lr_{}_batch_size_{}_fc1_{}_fc2_{}_fc3_{}_seed_{}".format(self.lr, self.batch_size, self.fc1, self.fc2, self.fc3, self.seed)
        pathname += "_clip_{}".format(config["clip"])
        pathname += "_tau_{}".format(config["tau"])
        now = datetime.now()    
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
        pathname += dt_string
        tensorboard_name = str(config["locexp"]) + '/runs/' + pathname
        self.writer = SummaryWriter(tensorboard_name)
        print("summery writer ", tensorboard_name)
        self.average_prediction = deque(maxlen=100)
        self.average_same_action = deque(maxlen=100)
        self.all_actions = []
        for a in range(self.action_size):
            action = torch.Tensor(1) * 0 +  a
            self.all_actions.append(action.to(self.device))
        self.best_r = 0
        self.best_q = 0
        self.best_r_step = 0
        self.best_q_step = 0
        self.steps = 0
    
    def pretrain(self, memory_ex):
        logging.debug("--------------------------pretrain update {}-----------------------------------------------".format(self.steps))
        states, next_states, actions = memory_ex.expert_policy(self.batch_size)
        states = states.type(torch.float32).div_(255)
        states = self.encoder.create_vector(states) # .detach()
        self.state_action_frq(states, actions)
        self.steps += 1
    
    def learn(self, memory_ex, memory_all):
        self.steps += 1
        states, next_states, actions = memory_ex.expert_policy(self.batch_size)
        states = states.type(torch.float32).div_(255)
        states = self.encoder.create_vector(states) #.detach()
        self.state_action_frq(states, actions)
        states, next_states, actions = memory_all.expert_policy(self.batch_size)
        states = states.type(torch.float32).div_(255)
        states = self.encoder.create_vector(states)
        next_states = next_states.type(torch.float32).div_(255)
        next_states = self.encoder.create_vector(next_states)
        self.compute_shift_function(states.detach(), next_states, actions)
        self.compute_r_function(states.detach(), actions)
        self.compute_q_function(states.detach(), next_states, actions)
        self.soft_update(self.R_local, self.R_target, self.tau)
        self.soft_update(self.q_shift_local, self.q_shift_target, self.tau)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
         
        return
            
    
    
    def compute_q_function(self, states, next_states, actions):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        actions = actions.type(torch.int64)
        #actions.requires_grad_(True)
        #print("action grad ", actions.requires_grad)
        # Get max predicted Q values (for next states) from target model
        if self.double_dqn:
            q_values = self.qnetwork_local(next_states).detach()
            _, best_action = q_values.max(1)
            Q_targets_next = self.qnetwork_target(next_states).detach()
            Q_targets_next = Q_targets_next.gather(1, best_action.unsqueeze(0)).squeeze(0).unsqueeze(1)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        # Get expected Q values from local model
        # Compute loss
        rewards = self.R_target(states).detach().gather(1, actions.detach()).squeeze(0)
        Q_targets = rewards + (self.gamma * Q_targets_next)
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q_expected, Q_targets.detach())
        
        # Get max predicted Q values (for next states) from target model
        
        self.writer.add_scalar('Q_loss', loss, self.steps)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        

    def compute_shift_function(self, states, next_states, actions):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        actions = actions.type(torch.int64)
        with torch.no_grad():
            # Get max predicted Q values (for next states) from target model
            #if self.double_dqn:
            #qt = self.q_shift_local(next_states)
            #max_q, max_actions = qt.max(1)
            #Q_targets_next = self.qnetwork_target(next_states).gather(1, max_actions.unsqueeze(1))
            #else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            Q_targets = self.gamma * Q_targets_next 

        # Get expected Q values from local model
        Q_expected = self.q_shift_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.optimizer_shift.zero_grad()
        loss.backward()
        self.writer.add_scalar('Shift_loss', loss, self.steps)
        self.optimizer_shift.step()
    
    
    def compute_r_function(self, states, actions, debug=False, log=False):
        actions = actions.type(torch.int64)
        # sum all other actions
        # print("state shape ", states.shape)
        size = states.shape[0]
        idx = 0
        all_zeros = [1 for i in range(actions.shape[0])]
        zeros = False
        self.predicter.eval()
        y_shift = self.q_shift_target(states).gather(1, actions).detach()
        log_a = self.get_action_prob(states, actions).detach()
        y_r_part1 = log_a - y_shift
        #y_r_part1 = log_a
        y_r_part2 = torch.empty((size, 1), dtype=torch.float32).to(self.device)
        for a, s in zip(actions, states):
            y_h = 0
            taken_actions = 0
            for b in self.all_actions:
                b = b.type(torch.int64).unsqueeze(1)
                n_b = self.get_action_prob(s.unsqueeze(0), b)
                if torch.eq(a, b) or n_b is None:
                    continue
                taken_actions += 1
                y_s = self.q_shift_target(s.unsqueeze(0)).detach().gather(1, b).item()
                n_b = n_b.data.item() - y_s
                #n_b = n_b.data.item()
                r_hat = self.R_target(s.unsqueeze(0)).gather(1, b).item()
                y_h += (r_hat - n_b)
                if log:
                    text = "a {} r _hat {:.2f} - n_b  {:.2f} | sh {:.2f} ".format(b.item(), r_hat, n_b, y_s)
                    print(text)
                    logging.debug(text)
            if taken_actions == 0:
                all_zeros[idx] = 0
                zeros = True
                y_r_part2[idx] = 0.0
            else:
                y_r_part2[idx] = (1. / taken_actions) * y_h
            idx += 1
            y_r = y_r_part1 + y_r_part2
        # check if there are zeros (no update for this tuble) remove them from states and
        if zeros:
            #print(all_zeros)
            #print(states)
            #print(actions)
            mask = torch.BoolTensor(all_zeros)
            states = states[mask]
            actions = actions[mask]
            y_r = y_r[mask]

        y = self.R_local(states).gather(1, actions)
        if log and not zeros:
            text = "Action {:.2f} r target {:.2f} =  n_a {:.2f} + n_b {:.2f}  y {:.2f}".format(actions[0].item(), y_r[0].item(), y_r_part1[0].item(), y_r_part2[0].item(), y[0].item()) 
            print(text)
            logging.debug(text)
            return


        r_loss = F.mse_loss(y, y_r.detach())

        # sys.exit()
        # Minimize the loss
        self.optimizer_r.zero_grad()
        r_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.R_local.parameters(), 5)
        self.optimizer_r.step()
        self.writer.add_scalar('Reward_loss', r_loss, self.steps)
        self.predicter.train()

    
    
    
    def get_action_prob(self, states, actions):
        """
        """
        actions = actions.type(torch.long)
        # check if action prob is zero
        output = self.predicter(states)
        output = F.softmax(output, dim=1)
        action_prob = output.gather(1, actions)
        action_prob = action_prob + torch.finfo(torch.float32).eps
        # check if one action if its to small
        if action_prob.shape[0] == 1:
            if action_prob.cpu().detach().numpy()[0][0] < 1e-4:
                return None
        action_prob = torch.log(action_prob)
        action_prob = torch.clamp(action_prob, min= self.clip, max=0)
        return action_prob


    def state_action_frq(self, states, action):
        """ Train classifer to compute state action freq
        """
        self.predicter.train()
        #output = self.predicter(states, train=True)
        output = self.predicter(states)
        output = output.squeeze(0)
        # logging.debug("out predicter {})".format(output))

        y = action.type(torch.long).squeeze(1)
        #print("y shape", y.shape)
        loss = nn.CrossEntropyLoss()(output, y)
        self.optimizer_pre.zero_grad()
        self.encoder_optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.predicter.parameters(), 1)
        self.optimizer_pre.step()
        self.encoder_optimizer.step()
        self.writer.add_scalar('Predict_loss', loss, self.steps)
        self.predicter.eval()


    
    def test_predicter(self, memory):
        """

        """
        self.predicter.eval()
        same_state_predition = 0
        for i in range(memory.idx):
            states = memory.obses[i]
            actions = memory.actions[i]
        
            states = torch.as_tensor(states, device=self.device).unsqueeze(0)
            actions = torch.as_tensor(actions, device=self.device)
            states = states.type(torch.float32)
            states = self.encoder.create_vector(states.detach())
            output = self.predicter(states)   
            output = F.softmax(output, dim=1)
            #print("state ", output.data)
            # create one hot encode y from actions
            y = actions.type(torch.long).item()
            p = torch.argmax(output.data).item()
            #print("a {}  p {}".format(y, p))
            text = "r  {}".format(self.R_local(states.detach()).detach()) 
            #print(text)
            if y==p:
                same_state_predition += 1
        text = "Same prediction {} of {} ".format(same_state_predition, memory.idx)
        print(text)
        logging.debug(text)




    def soft_update(self, local_model, target_model, tau=4):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        # print("use tau", tau)
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def load(self, filename):
        self.predicter.load_state_dict(torch.load(filename + "_predicter.pth"))
        self.optimizer_pre.load_state_dict(torch.load(filename + "_predicter_optimizer.pth"))
        self.R_local.load_state_dict(torch.load(filename + "_r_net.pth"))
        self.qnetwork_local.load_state_dict(torch.load(filename + "_q_net.pth"))
        self.encoder.load_state_dict(torch.load(filename + "_encoder.pth"))
        print("Load models to {}".format(filename))


    def save(self, filename):
        """
        """
        mkdir("", filename)
        torch.save(self.predicter.state_dict(), filename + "_predicter.pth")
        torch.save(self.optimizer_pre.state_dict(), filename + "_predicter_optimizer.pth")
        torch.save(self.qnetwork_local.state_dict(), filename + "_q_net.pth")
        torch.save(self.optimizer.state_dict(), filename + "_q_net_optimizer.pth")
        torch.save(self.R_local.state_dict(), filename + "_r_net.pth")
        torch.save(self.optimizer_r.state_dict(), filename + "_r_net_optimizer.pth")
        torch.save(self.q_shift_local.state_dict(), filename + "_q_shift_net.pth")
        torch.save(self.optimizer_shift.state_dict(), filename + "_q__shift_net_optimizer.pth")
        torch.save(self.encoder.state_dict(), filename + "_encoder.pth")
        torch.save(self.encoder_optimizer.state_dict(), filename + "_endcoder_optimizer.pth")
        print("save models to {}".format(filename))

    def test_q_value(self, memory):
        same_r = 0
        same_q = 0
        same_sh = 0
        test_elements = memory.idx
        all_diff = 0
        error = True
        for i in range(test_elements):
            states = memory.obses[i]
            actions = memory.actions[i]
            states = torch.as_tensor(states, device=self.device).unsqueeze(0)
            states = states.type(torch.float32)
            states = self.encoder.create_vector(states.detach())
            actions = torch.as_tensor(actions, device=self.device)
            # print("test", states)
            with torch.no_grad():
                r_values = self.R_local(states.detach()).detach()
                q_values = self.qnetwork_local(states.detach()).detach()
                qsh_values = self.q_shift_target(states.detach()).detach()
                best_sh = torch.argmax(qsh_values).item()
                best_r = torch.argmax(r_values).item()
                best_q = torch.argmax(q_values).item()
                text = "action {} r value {} ".format(actions.item(), r_values.data)
                # self.compute_r_function(states.detach(), actions.unsqueeze(0), log=True)
                logging.debug(text)
                #text = "action {} q value {} ".format(actions.item(), q_values.data)
                #logging.debug(text)
                # print(text)
                actions = actions.type(torch.int64)
                if  actions.item() == best_q:
                    same_q += 1
                if  actions.item() == best_r:
                    same_r += 1

                if  actions.item() == best_sh:
                    same_sh += 1

        text = "same r {} of {} ".format(same_r, test_elements)
        print(text)
        logging.debug(text)
        text = "same q {} of {} ".format(same_q, test_elements)
        print(text)
        logging.debug(text)
        text = "same q shift {} of {} ".format(same_q, test_elements)
        print(text)
        logging.debug(text)
        self.writer.add_scalar('same_r', same_r, self.steps)
        self.writer.add_scalar('same_q', same_q, self.steps)
        
        if same_r > self.best_r:
            self.best_r = same_r
            self.best_r_step = self.steps

        if same_q > self.best_q:
            self.best_q = same_q
            self.best_q_step = self.steps

    def act(self, states):
        states = torch.as_tensor(states, device=self.device).unsqueeze(0)
        states = states.type(torch.float32).div_(255)
        states = self.encoder.create_vector(states)
        q_values = self.qnetwork_local(states.detach()).detach()
        action = torch.argmax(q_values).item()
        return action 

