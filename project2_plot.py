import numpy as np
import matplotlib.pyplot as plt
import json

def moving_average(a, n) :
    ma = [a[0]]
    for v in range(1,len(a)):
        ma.append(np.mean(a[:v][-1*n:]))
    return ma

exp1 = 'p2_out20180319-042344.json'
exp2 = 'p2_out20180318-233951.json'
exp3 = 'p2_out20180318-182345.json'
exp4 = 'p2_out20180318-205024.json'

with open(exp1, 'r') as f:
    d = json.load(f)
    print(d.keys())

    ma = moving_average(d['rewards'],100)
    print(max(ma))
    plt.plot(list(range(5000)), ma)
    #plt.plot(list(range(100)), d['trial_rewards'])
    #plt.axvline(x=3615,color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.title('Training Performance')
    plt.show()