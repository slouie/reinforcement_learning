
def kStep(step, probToState, valueEstimates, rewards):
    if step == 0:
        return valueEstimates[0] + 1.0*(probToState * rewards[0]
                                        + probToState * valueEstimates[1]
                                        + (1.0 - probToState) * rewards[1]
                                        + (1.0 - probToState) * valueEstimates[2]
                                        - valueEstimates[0])
    else:
        discountedRewards = probToState * (rewards[0]+rewards[2]) + (1.0-probToState) * (rewards[1]+rewards[3])
        for x in xrange(step-1):
            discountedRewards += ( rewards[x+4] if (x+4) < 7 else 0 )
        return valueEstimates[0] + discountedRewards + ( valueEstimates[step+2] if step < 5 else 0 )

def calculateLambda(probToState, valueEstimates, rewards):
    kEstimates = []
    steps = 6
    for step in xrange(steps):
        kEstimates.append( kStep(step, probToState, valueEstimates, rewards) )
    kEstimates.reverse()
    #print kEstimates
    exp = range(steps-1,-1,-1)
    terms = []
    for x in xrange(steps-1):
        terms.append("({})*x^({})".format(kEstimates[x+1], exp[x+1]))
    return "{}={}*x^5 + ".format(kEstimates[0],kEstimates[0]) + "(1-x)*(" + "+".join(terms) + ")"

# print calculateLambda(probToState=0.5, valueEstimates=[0,3,8,2,1,2,0], rewards=[0,0,0,4,1,1,1])
# print calculateLambda(probToState=0.81, valueEstimates=[0.0,4.0,25.7,0.0,20.1,12.2,0.0], rewards=[7.9,-5.1,2.5,-7.2,9.0,0.0,1.6])
# print calculateLambda(probToState=0.22,valueEstimates=[0.0,-5.2,0.0,25.4,10.6,9.2,12.3],rewards=[-2.4,0.8,4.0,2.5,8.6,-6.4,6.1])
# print calculateLambda(probToState=0.64,valueEstimates=[0.0,4.9,7.8,-2.3,25.5,-10.2,-6.5],rewards=[-2.4,9.6,-7.8,0.1,3.4,-2.1,7.9])

print calculateLambda(probToState=0.63,valueEstimates=[0.0,0.0,3.7,24.9,-0.7,-4.3,14.8],rewards=[-3.3,7.9,-1.5,-4.7,2.4,3.5,-1.7])
print calculateLambda(probToState=0.35, valueEstimates=[0.0,0.0,0.0,11.6,22.1,11.5,0.0], rewards=[-3.9,0.2,-3.4,0.0,-2.5,3.2,3.8])
print calculateLambda(probToState=0.22, valueEstimates=[0.0,16.9,10.5,14.5,-3.8,0.0,2.9], rewards=[6.8,-1.1,2.0,-3.2,6.8,8.1,-1.8])
print calculateLambda(probToState=0.55, valueEstimates=[0.0,-2.9,0.0,10.0,4.4,16.5,18.4], rewards=[-2.1,2.7,8.7,-1.2,-0.1,-2.3,2.2])
print calculateLambda(probToState=0.64, valueEstimates=[0.0,0.0,9.3,0.0,21.8,5.0,11.5], rewards=[-4.0,-2.2,3.8,6.5,-2.8,-4.1,7.4])
print calculateLambda(probToState=0.36, valueEstimates=[0.0,18.3,5.2,17.2,8.9,0.0,23.2], rewards=[7.7,0.0,8.2,9.3,-1.0,-2.1,6.6])
print calculateLambda(probToState=0.73, valueEstimates=[0.0,12.4,0.0,0.0,0.0,5.2,-2.4], rewards=[5.9,-4.3,-1.6,9.3,-1.3,2.4,3.1])
print calculateLambda(probToState=0.82, valueEstimates=[0.0,8.5,0.0,6.9,0.0,9.6,0.0], rewards=[-1.8,6.8,3.6,0.0,1.2,5.6,-1.5])
print calculateLambda(probToState=0.71, valueEstimates=[0.0,0.0,24.8,20.1,0.0,17.6,0.0], rewards=[4.4,5.5,1.5,1.9,2.7,2.3,3.5])
print calculateLambda(probToState=0.82, valueEstimates=[0.0,7.6,7.1,20.3,0.0,13.7,5.3], rewards=[7.2,-0.2,4.7,0.0,9.8,-0.4,-1.2])