import numpy as np
import data_module as dm
import network_module as nn
import matplotlib.pyplot as plt

tau = 40000
N=50
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.05, 1.05
M, D , etha = 3, 2, 7.8e-3

generator = dm.DataModule(x_min, x_max, N)
X_train, Y_train = generator.gen_data(generator.sinusodial)
n_sample = generator.get_sample_number()

network = nn.SimpleNetwork(M, D, etha)
W10, W20 = network.learn(X_train, Y_train, tau)
Y_regress = network.f_2D_regression(X_train)
Z_regress = network.z1_2D_regression(X_train)

print(network.error_rate(X_train, Y_train))

color = ['ORANGE', "GREEN", "MAGENTA"]
for m in range(3):
    plt.plot(X_train.reshape(-1).tolist(), Z_regress[:, m].reshape(-1).tolist(), c=color[m])
plt.plot(X_train.reshape(-1).tolist(), Y_regress, c='RED')
plt.scatter(X_train, Y_train, c='BLUE', s=4)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
