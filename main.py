# follow along for neural nine video https://www.youtube.com/watch?v=MSrfaI1gGjI

import random
import gym
import numpy as np

env = gym.make('Taxi-v3')
alpha = 0.9 # learning rate
gamma = 0.95 # 0 for short term, 1 for long term learning, called the discount factor
epsilon = 1 # randomness, exploration rate, 0 always in the Q, 1 always random
epsilon_decay = 0.995
min_epsilon = 0.01 # leave room for randomness

num_episodes = 10000 # how many times the agent will play the game
max_steps = 100 # exhaust moves, episode terminates

# for each possible state, you will have a value, 
# (this is not what i want, because in 3d games or physics based ones
# the numbers of states will be huge, let alone the number of actions the agent can take
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
  if random.uniform(0, 1) < epsilon:
    return env.action_space.sample()
  else:
    return np.argmax(q_table[state])
  
for episode in range(num_episodes):
  state, _ = env.reset()
  done = False
  
  for step in range(max_steps):
    action = choose_action(state)
    next_state, reward, done, truncated, info = env.step(action) # move one step forward

    old_value = q_table[state, action]
    next_value = np.max(q_table[next_state])

    q_table[state, action] = (1-alpha) * old_value + alpha * (reward + gamma * next_value)
    state = next_state

    if done or truncated:
      break

  epsilon = max(min_epsilon, epsilon * epsilon_decay)

env = gym.make('Taxi-v3', render_mode='human')
for episode in range(5):
  state, _ = env.reset()
  done = False

  print("Episode", episode)
  
  for step in range(max_steps):
    env.render()
    action = np.argmax(q_table[state])
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state

    if done or truncated:
      env.render()
      print("Finished episode", episode, 'with reward', reward)
      