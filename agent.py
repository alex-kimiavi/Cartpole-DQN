import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from params import Parameters


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent(nn.Module):
    def __init__(self, obs_space=4, action_space=2):
        super().__init__()
        self.input = nn.Linear(obs_space, 64)
        self.h1 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input(x))
        x = F.relu(self.h1(x))
        x = self.out(x)
        return x

class Replay():
    def __init__(self, size=100000):
        self.size = size 
        self.state, self.action, self.reward, self.done, self.next_state = deque(maxlen=size), deque(maxlen=size), deque(maxlen=size),deque(maxlen=size),deque(maxlen=size),
    
    def sample_batch(self, ind, obj) -> torch.Tensor:
        return torch.from_numpy(np.vstack([obj[i] for i in ind])).to(device)

    def append(self, state, action, reward, done, next_state) -> None:
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.next_state.append(next_state)
    
    def sample(self, batch_size) -> tuple[torch.Tensor,...]:
        
        ind = np.random.random_integers(low=0, high=len(self.state) - 1, size=batch_size)
        return (
            self.sample_batch(ind, self.state),
            self.sample_batch(ind, self.action), 
            self.sample_batch(ind, self.reward), 
            self.sample_batch(ind, self.done), 
            self.sample_batch(ind, self.next_state),
        )

    def __len__(self):
        return len(self.state)


def process_state(state : np.array):
    return torch.from_numpy(np.array([state]))


def train_batch(agent: Agent, optimizer: torch.optim.Optimizer, replay: Replay, params: Parameters) -> float:
    states, actions, rewards, done, next_states = replay.sample(params.batch_size)
    rewards = rewards.squeeze()
    done = done.type(torch.int32).squeeze()
    done = 1 - done
    actions = actions.long()
    y: torch.Tensor = rewards + params.gamma*torch.mul(done,torch.max(agent(next_states), dim=1).values)
    state_q_vals = torch.gather(agent(states), dim=1, index=actions).squeeze()
    loss = (y-state_q_vals).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach().cpu().numpy()
