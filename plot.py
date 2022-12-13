import numpy as np
import matplotlib.pyplot as plt

data128 = np.load('models/flowTorch_16_128_32/mean_loss.npy')
data256 = np.load('models/flowTorch_16_128_32/mean_loss.npy')
data512 = np.load('models/flowTorch_16_128_32/mean_loss.npy')
iteration = [data128[::2], data256[::2], data512[::2]]
mean_loss = [data128[1::2], data256[1::2], data512[1::2]]
label = ['128', '256', '512']

plt.title('Training Mean Loss (average 100 iterations)')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')

for i in range(len(iteration)):
    plt.plot(iteration[i], mean_loss[i], label='points=%s, emb=32, batch=16' % label[i])

plt.legend()
plt.show()