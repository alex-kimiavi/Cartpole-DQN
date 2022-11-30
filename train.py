import gym
import torch 
import numpy as np
from agent import Agent, Replay, process_state, train_batch
from params import Parameters


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    params = Parameters()
    env = gym.make('CartPole-v1')
    agent = Agent(obs_space=4, action_space=2).to(device)
    replay = Replay(params.replay_length)

    optimizer = torch.optim.Adam(params=agent.parameters(), lr=params.learning_rate)
    eps = params.eps_max
    steps = 0 
    for episode in range(params.num_episodes):
        done = False
        state = process_state(env.reset()[0])
        episode_reward = 0 
        loss = 0
        while not done:
            eps = params.eps_min if steps > params.replay_length else eps - params.eps_decay
            steps += 1
            action = env.action_space.sample() if np.random.rand() < eps else torch.argmax(agent(state.to(device)).squeeze()).cpu().numpy()
            
            next_state, reward, done, _, _ = env.step(action)
            next_state = process_state(next_state)
            replay.append(state, action, reward, done, next_state)
            episode_reward += reward
            state = next_state

            if steps > params.pre_train_steps and steps % params.update_frequency == 0:
                loss = train_batch(agent, optimizer, replay, params)
        if episode % 50 == 0:
            torch.save(agent, 'agent.dump')
        
        with open('log.txt', 'a') as f:
            f.write(f'{episode} {episode_reward}\n') 
        print(episode, steps, loss, episode_reward, eps, agent(state.to(device)).detach().cpu().numpy())
