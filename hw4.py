import gym
import numpy as np
import random

env = gym.make('Taxi-v2')
# env.seed(0)
env.reset()
env.render()

Q = np.zeros([env.observation_space.n, env.action_space.n])

G = 0.0
alpha = 1.0
gamma = 0.9

for episode in range(1, 10000):
    done = False
    G, reward = 0.0, 0.0
    state = env.reset()

    while not done:
        if episode < 1100:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        state2, reward, done, info = env.step(action)
        if not done:
            Q[state,action] += alpha*(reward + gamma*np.max(Q[state2]) - Q[state,action])
        else:
            Q[state,action] += alpha*(reward - Q[state,action])
        G += reward
        state = state2

    if episode % 1000 == 0:
        #print "Episode {}, Total Reward {}".format( episode, G )
        #print Q[462][4],Q[398][3],Q[253][0],Q[377][1],Q[83][5]
        print(Q[69][2])

    # if (( abs( Q[462][4] + 11.374402515 ) < 0.0000000001 ) and
    #     ( abs( Q[398][3] - 4.348907 ) < 0.0000000001 ) and
    #     ( abs( Q[253][0] + 0.5856821173 ) < 0.0000000001 ) and
    #     ( abs( Q[377][1] - 9.683 ) < 0.0000000001 ) and
    #     ( abs( Q[83][5] + 12.8232660372 ) < 0.0000000001 )):
    #     print "Converged"
    #     break
print(Q[462][4],Q[398][3],Q[253][0],Q[377][1],Q[83][5])

print(Q[483][5])
print(Q[391][1])
print(Q[69][2])
print(Q[132][1])
print(Q[379][4])
print(Q[499][2])
print(Q[376][2])
print(Q[482][4])
print(Q[462][5])
print(Q[103][3])
