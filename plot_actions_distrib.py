import numpy as np

path = "actions.npy"
actions = np.load(path, allow_pickle=True)
# actions shape : (100, 14) : 14 is the dimension of the action space
# plot the distribution of actions as a histogram

print(actions[10])