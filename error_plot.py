import data_module as dm
import network_module as nn
import numpy as np
import matplotlib.pyplot as plt

x_min, x_max = -1.0, 1.0
M_max, N, tau = 10, 50, 25000
etha = 7.8e-3
samp = 30

generator = dm.DataModule(x_min, x_max, N)
network = nn.SimpleNetwork(3, 2, etha)

f = generator.sinusodial

X_train, Y_train = generator.gen_data(f)
generator.set_sample_number(int(N*3/7))
X_val, Y_val = generator.gen_data(f)
#(debug code)X_val, Y_val = X_train, Y_train

plot_num = 4
if plot_num == 1:
    err = []
    #(debug code)for m in range(M_max-1, M_max):
    for m in range(1, M_max+1):
        err_m = []
        network.set_h_unit(m)
        print('now evaluate m=', m)
        for i in range(samp):
            #(debug code)print(1)
            network.learn(X_train, Y_train, tau)
            #(debug code)print(2)
            temp = network.error_rate(X_val, Y_val)
            print(temp)
            err_m.append(temp)
        err.append(err_m)

    for m in range(M_max):
        complexity = [m+1] * samp
        plt.scatter(complexity, err[m], marker='x', c='BLUE')
    plt.show()

if plot_num == 2:
    err =[]
    tau = 850
    etha0, step = 0.00001, 0.000005
    num_step = 200
    network.set_h_unit(3)
    etha_list = [etha0 + i * step for i in range(num_step)]

    switch = False
    for i in range(num_step):
        print('laerning rate: ', etha_list[i])
        network.set_learning_rate(etha_list[i])
        if switch == False:
            W1, W2 = network.learn(X_train, Y_train, tau)
            switch = True
        else:
            network.set_weight(W1, W2)
            network.learn(X_train, Y_train, tau)
        temp = network.error_rate(X_val, Y_val)
        print('error: ', temp)
        err.append(temp)

    plt.plot(etha_list, err, c='BLUE')
    plt.show()

if plot_num == 3:
    err_train = []
    err_val = []

    for m in range(4, M_max+1):
        print('Now evaluate m = ', m)
        network.set_h_unit(m)
        err_m_train = 0
        err_m_val = 0
        for i in range(samp):
            network.learn(X_train, Y_train, tau)
            temp_train = network.error_rate(X_train, Y_train)
            temp_val = network.error_rate(X_val, Y_val)
            print(temp_train, temp_val)
            err_m_train += temp_train
            err_m_val += temp_val
        err_train.append(err_m_train/samp)
        err_val.append(err_m_val/samp)
        print(err_train[m-1], err_val[m-1])

    complexity = [i for i in range(1, M_max+1)]
    plt.plot(complexity, err_train, c = 'RED')
    plt.plot(complexity, err_val, c = 'BLUE')
    plt.show()

if plot_num == 4:
    tau_max = 40000
    err_plot = []
    network.set_h_unit(4)
    for iter in range(200, tau_max, 200):
        network.set_h_unit(4)
        err_iter = 0
        print('iter step: ', iter)
        for i in range(samp):
            network.learn(X_train, Y_train, iter)
            temp = network.error_rate(X_train, Y_train)
            print(temp)
            err_iter += temp
        err_plot.append(err_iter/samp)
    plt.plot([iter for iter in range(200, tau_max, 200)], err_plot)
    plt.show()
