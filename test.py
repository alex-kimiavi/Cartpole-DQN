import torch 
import gym 
from agent import Agent, process_state



device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    agent : Agent = torch.load('agent.dump')
    
    
    while True:
        done = False 
        state = process_state(env.reset()[0])
        episode_reward = 0
        while not done and episode_reward < 1000:
            env.render()
            action =  torch.argmax(agent(state.to(device)).squeeze()).cpu().numpy()
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = process_state(next_state)
        print(episode_reward)
