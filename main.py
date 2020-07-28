# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:58:20 2020

@author: agutier4
"""
from unityagents import UnityEnvironment
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent import Agent

#env = UnityEnvironment(file_name='Reacher20/Reacher.exe', base_port=63457)
env = UnityEnvironment(file_name='Reacher20/Reacher.exe')
#env = UnityEnvironment(file_name='Reacher.exe', no_graphics=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
#policy = Policy(state_size, action_size, 0).to(device)



#env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
#states = env_info.vector_observations                  # get the current state (for each agent)
#scores = np.zeros(num_agents)                          # initialize the score (for each agent)

agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)

def ddpg(n_episodes=500, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] 
        states = env_info.vector_observations
        
        score = np.zeros(num_agents)
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states)
            #if (t % 50 == 0):
                #print(actions[0])
            #print("t is",t," and action:", action)
            #next_state, reward, done, _ = env.step(action)
            env_info = env.step(actions)[brain_name]               # send the action to the environment                            
            next_states = env_info.vector_observations               # get the next state        
            rewards = env_info.rewards                               # get the reward        
            dones = env_info.local_done                              # see if episode has finished  
            score += rewards
            for i in range(20) :
                agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])
            states = next_states
            
            if np.any(dones):
                break 
        scores_deque.append(np.mean(score))
        scores.append(score)
        #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        #print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, np.mean(score)), end="")
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
        if np.mean(scores_deque) >= 5 and len(scores_deque) == 100:
            print('\rSolved in {} episodes, with mean score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return scores
    
scores = ddpg()

mean_scores = np.mean(scores, axis = 1)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), mean_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()