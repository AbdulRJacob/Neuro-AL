import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import deque
from datasets.CLEVR.CLEVR import CLEVRHans
import pipelines.clevr as clevr


class ReplayBuffer():
    def __init__(self, size:int):
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    return int(torch.argmax(dqn(state)))

def epsilon_greedy_action(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
   
    q_values = dqn(state.to(dtype=torch.float))
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)

def update_target(target_dqn:DQN, policy_dqn:DQN):
    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
    
    bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()





class ABAEnv:
    def __init__(self, num_slots: int , num_pred: int, num_assump: int, num_labels: int ):
        self.state_size = num_slots * num_pred + 2 * num_assump + num_labels
        self.action_size = 2 * num_assump + num_labels
        self.label_idx = num_labels
        self.assump_idx = num_assump
        self.state = torch.zeros(self.state_size)

    def state_to_index(state):
        # Convert the binary state tensor to a list of integers
        state_list = state.to(dtype=torch.int).tolist()
        # Join the list elements to form a binary string
        binary_string = ''.join(map(str, state_list))
        # Convert the binary string to an integer
        index = int(binary_string, 2)
        return index

    def init_state(self,slots):
        t_slots = []
        for slot in slots:
            if slot[-1] == 0:
                t_slots.append(torch.zeros(15))
                continue

            slot = slot[3:]
            slot = slot[:-1]
            t_slots.append(torch.from_numpy(slot))

        t_slots.append(torch.zeros(self.action_size))

        t_slots = torch.cat(t_slots, dim=0)

        self.state = t_slots

    def check_assumption_inconsistency(self,array):
        n = self.assump_idx
        half_n = n // 2

        if n < half_n * 2:
            raise ValueError("Array size must be even")

        # Create a mask to identify elements at index n and n + n/2
        mask_n = array[:-half_n]
        mask_n_plus_half = array[half_n:]

        # Check if both elements are 1
        result = (mask_n == 1) & (mask_n_plus_half == 1)

        # Return True if at least one pair contains both values being 1
        return torch.any(result)


    def terminal_state(self):
        label_states = torch.flip(self.state, dims=[0])[:self.label_idx]

        
        return torch.any(label_states)

    def step(self, action,target):
        action = self.state_size  - self.action_size + action
        # Action is an integer representing the index of the property to toggle
        self.state[action] = 1 - self.state[action]  # Toggle the truth value of the property
        reward = self.compute_reward(np.argmax(target))  # Compute reward based on the updated state
        return self.state, reward

    def compute_reward(self,target):
        torch.flip(self.state, dims=[0])
        label_states = torch.flip(self.state, dims=[0])[:self.label_idx]

        if label_states[target] == 1:
            return 1
        if torch.any(label_states):
            return -10
        
        assump_states = self.state[self.state_size - self.action_size:self.assump_idx]
        if self.check_assumption_inconsistency(assump_states):
            return -5
        
        return -0.5
         

    def reset(self):
        # Reset the environment to its initial state
        self.state = torch.zeros(self.state_size)
        return self.state

class ClassifierAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Initialize DQN model
        self.dqn = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()




    def update_policy(self, state, action, reward, next_state):


        state = torch.stack(state).to(dtype=torch.float)
        next_state = torch.stack(next_state).to(dtype=torch.float)
        action_tensor = torch.tensor(action)
        reward = torch.tensor(reward)

        action_onehot = torch.zeros(len(action_tensor), self.action_dim)

        action_onehot.scatter_(1, action_tensor.unsqueeze(1), 1)
    



        # Compute Q-values for current state and next state
        current_q_values = self.dqn(state)
        next_q_values = self.dqn(next_state)
        
        
        # Compute Q-value for the selected action
        current_q_values_selected = current_q_values.gather(1, action_tensor.unsqueeze(1))

        # Compute target Q-value using the Bellman equation
        max_next_q_value = torch.max(next_q_values, dim=1)[0]  # Taking the maximum Q-value across actions
        target_q_value = reward + self.gamma * max_next_q_value

        # Compute loss between current and target Q-values
        loss = self.criterion(current_q_values_selected, target_q_value.unsqueeze(1))
                
        # Perform backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main(model, dataset):
    # Define environment and agent
    env = ABAEnv(num_slots=11, num_pred=15, num_assump=5, num_labels=3)
    agent = ClassifierAgent(state_dim=env.state_size, action_dim=env.action_size,
                            learning_rate=0.0025, gamma=0.99)

    # Define replay buffer
    replay_buffer = ReplayBuffer(size=1000)

    num_episodes = len(dataset)
    max_steps_per_episode = 13
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.95
    batch_size = 64

    for episode in range(num_episodes):
        pred = dataset[episode]["target"]
        print(pred)
        target = dataset[episode]["class"]
        env.init_state(pred)
        state = env.state

        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        for step in range(max_steps_per_episode):
            action =  epsilon_greedy_action(epsilon,agent.dqn,state)#agent.select_action(state, epsilon)
            next_state, reward = env.step(action, target)  # Take action and get next state, reward

            # Store transition in replay buffer
            replay_buffer.push((state, action, reward, next_state))

            # Sample from replay buffer
            if len(replay_buffer.buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)
                
                # Update agent's policy using the sampled transitions
                agent.update_policy(state_batch, action_batch, reward_batch, next_state_batch)

            state = next_state 

            if env.terminal_state():
                break

    model_save_path = 'model2.pt'
    torch.save(agent.dqn.state_dict(), model_save_path)


def get_prediction(dataset):
    state_dim = 178  # Replace with actual state dimension
    action_dim = 13  # Replace with actual action dimension
    loaded_agent = ClassifierAgent(state_dim=state_dim, action_dim=action_dim, learning_rate=0.001, gamma=0.99)

    # Load the saved state dictionary into the model
    loaded_agent.dqn.load_state_dict(torch.load('model2.pt'))

    # Set the model to evaluation mode
    loaded_agent.dqn.eval()
    env = ABAEnv(num_slots=11, num_pred=15, num_assump=5, num_labels=3)
    prediction = []
    true = []

    for i in range(len(dataset)):
        pred = dataset[i]["target"]
        target = dataset[i]["class"]
        env.init_state(pred)
        state = env.state

        state = torch.tensor(state).float()  # Replace `your_state` with the actual state data

        steps = 0
        while not env.terminal_state():
            if steps > 20:
                break
            q_values = loaded_agent.dqn(state)
            action = torch.argmax(q_values).item()
            next_state, reward = env.step(action, target)
            state = torch.tensor(next_state).unsqueeze(0).float()
            steps += 1

        if action == 10:
            prediction.append(2)
        elif action == 11:
            prediction.append(1)
        elif action == 12:
            prediction.append(0)
        else:
            prediction.append(0)

        true.append(np.argmax(target))
    return prediction, true

if __name__ == "__main__":

    nal = clevr.get_clevr_nal_model()
    transform = transforms.Compose(
            [
                transforms.Resize((64, 64), antialias=None),
                transforms.PILToTensor(), 
            ])
    dataset = CLEVRHans(transform=transform)
    loader = DataLoader(dataset,len(dataset),num_workers=2,shuffle=True)
    
    # main(nal,dataset)
        
