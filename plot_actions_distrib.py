import numpy as np

path = "actions.npy"
actions = np.load(path, allow_pickle=True)
# actions shape : (100, 14) : 14 is the dimension of the action space
# plot the distribution of actions as a histogram
flattened_actions = actions.flatten()

import matplotlib.pyplot as plt
plt.hist(flattened_actions, bins=100)
plt.xlabel('Action Value')
plt.ylabel('Frequency')
plt.title('Distribution of Actions')
plt.grid()
plt.show()

