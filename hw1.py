import mdptoolbox
import numpy as np

def transitionTable(N, isBadSide):
    prob = np.zeros((2, 8*N, 8*N))
    for row in xrange(8*N):
        goodSides = np.array([row+idx+1 for idx, side in enumerate(isBadSide) if side == 0])
        goodSides = goodSides[goodSides<8*N]
        prob[0][row][goodSides] = 1.0/N
        prob[0][row][-1] = 1.0 - sum(prob[0][row][:-1])
        prob[1][row][-1] = 1.0
    return prob

def rewardTable(N, isBadSide):
    reward = np.zeros((8*N, 2))
    reward[:, 0] = 0
    reward[:, 1] = xrange(8*N)
    reward[-1,1] = 0
    return reward

def expectedValue(N, isBadSide):
    prob = transitionTable(N, isBadSide)
    rewards = rewardTable(N, isBadSide)

    vi = mdptoolbox.mdp.ValueIteration(prob, rewards, 1.0)
    vi.run()
    print "N={}\nisBadSide={}\nExpected={}".format(N,isBadSide,vi.V[0])


# expectedValue(21,[1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0])
# expectedValue(22,[1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0])
# expectedValue(6,[1,1,1,0,0,0])

expectedValue(11,[0,1,1,1,0,1,0,0,1,0,0])
expectedValue(9,[0,0,1,0,0,0,0,1,0])
expectedValue(20,[0,1,1,0,1,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0])
expectedValue(5,[0,1,0,1,1])
expectedValue(12,[0,0,0,0,1,0,1,0,0,0,0,0])
expectedValue(26,[0,0,0,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0,0])
expectedValue(28,[0,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,1])
expectedValue(10,[0,1,0,1,0,1,0,0,0,0])
expectedValue(19,[0,1,0,1,1,1,1,0,1,0,1,0,0,0,0,0,0,0,0])
expectedValue(12,[0,0,0,0,0,0,0,0,1,1,1,1])