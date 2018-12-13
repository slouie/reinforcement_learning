from cvxopt import matrix, solvers

# V     - P + S <= 0
# V + R     - S <= 0
# V - R + P     <= 0

# - R <= 0
# - P <= 0
# - S <= 0

#     R + P + S <= 1
#   - R - P - S <= -1

inputs = [
    #[[0.0,1.0,-1.0],[-1.0,0.0,1.0],[1.0,-1.0,0.0]],
    #[[0.0,2.0,-1.0],[-2.0,0.0,1.0],[1.0,-1.0,0.0]],
    # [[0.0, 4.98, -0.51], [-4.98, 0.0, 0.72], [0.51, -0.72, 0.0]],
    # [[0.0, 1.48, -1.0], [-1.48, 0.0, 3.64], [1.0, -3.64, 0.0]],
    # [[0.0, 3.05, -1.0], [-3.05, 0.0, 1.31], [1.0, -1.31, 0.0]],
    # [[0.0, 1.0, -0.92], [-1.0, 0.0, 2.39], [0.92, -2.39, 0.0]],
    # [[0.0, 2.63, -4.9], [-2.63, 0.0, 1.0], [4.9, -1.0, 0.0]],
    [[0.0, 1.0, -1.0], [-1.0, 0.0, 3.51], [1.0, -3.51, 0.0]],
    [[0.0, 1.62, -4.18], [-1.62, 0.0, 1.92], [4.18, -1.92, 0.0]],
    [[0.0, 1.0, -1.91], [-1.0, 0.0, 4.1], [1.91, -4.1, 0.0]],
    [[0.0, 3.84, -1.0], [-3.84, 0.0, 1.0], [1.0, -1.0, 0.0]],
    [[0.0, 4.96, -3.81], [-4.96, 0.0, 1.0], [3.81, -1.0, 0.0]]

]
for input in inputs:
    A = [[1.] + row for row in input]
    A.append([0., -1., 0., 0.])
    A.append([0., 0., -1., 0.])
    A.append([0., 0., 0., -1.])
    A.append([0., 1., 1., 1.])
    A.append([0., -1., -1., -1.])
    print(A)
    A = matrix(A).ctrans()
    c = matrix([-1.,0.,0.,0.])
    b = matrix([0.,0.,0.,0.,0.,0.,1.,-1.])
    sol = solvers.lp(c, A, b)
    print(sol['x'])

