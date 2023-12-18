import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import re
from torch import nn
mse = nn.MSELoss()


# 真正的函数
def f_gt(u, grav):
    return (u**2 * (1 - grav * (1-u)**4))/(u**2 + 0.5 * (1-u)**4)

def f_pf(u, beta, coe, power_nb):
    if power_nb == 1:
        res = coe[0] + coe[1] * u + beta * (coe[2] + coe[3] * u)
    elif power_nb == 2:
        res = coe[0] + coe[1] * u + coe[2] * u **2 + beta * (coe[3] + coe[4] * u + coe[5] * u **2)
    elif power_nb == 3:
        res = coe[0] + coe[1] * u + coe[2] * u **2 + coe[3] * u **3 \
              + beta * (coe[4] + coe[5] * u + coe[6] * u **2 + coe[7] * u **3)
    elif power_nb == 4:
        res = coe[0] + coe[1] * u + coe[2] * u **2 + coe[3] * u **3 + coe[4] * u **4 \
              + beta * (coe[5] + coe[6] * u + coe[7] * u **2 + coe[8] * u **3 + coe[9] * u **4)
    elif power_nb == 5:
        res = coe[0] + coe[1] * u + coe[2] * u **2 + coe[3] * u **3 + coe[4] * u **4 + coe[5] * u **5 \
              + beta * (coe[6] + coe[7] * u + coe[8] * u **2 + coe[9] * u **3 + coe[10] * u **4 + coe[11] * u **5)
    elif power_nb == 6:
        res = coe[0] + coe[1] * u + coe[2] * u **2 + coe[3] * u **3 + coe[4] * u **4 + coe[5] * u **5 + coe[6] * u **6 \
              + beta * (coe[7] + coe[8] * u + coe[9] * u **2 + coe[10] * u **3 + coe[11] * u **4 + coe[12] * u **5 + coe[13] * u **6)
    elif power_nb == 7:
        f = coe[0] + coe[1] * u + coe[2] * u **2 + coe[3] * u **3 + coe[4] * u **4 + coe[5] * u **5 + coe[6] * u **6 + coe[7] * u **7 \
            + beta * (coe[8] + coe[9] * u + coe[10] * u **2 + coe[11] * u **3 + coe[12] * u **4 + coe[13] * u **5 + coe[14] * u **6 + coe[15] * u **7 )
        return f
    elif power_nb == 8:
        f = coe[0] + coe[1] * u + coe[2] * u **2 + coe[3] * u **3 + coe[4] * u **4 + coe[5] * u **5 + coe[6] * u **6 + coe[7] * u **7 + coe[8] * u **8 \
            + beta * (coe[9] + coe[10] * u + coe[11] * u **2 + coe[12] * u **3 + coe[13] * u **4 + coe[14] * u **5 + coe[15] * u **6 + coe[16] * u **7 + coe[17] * u **8)
        return f
    return res

def plot_real_and_predict_function(x, y_gt, y_pd, name='name'):
    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    plot_1, = plt.plot(x, y_gt, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x, y_pd, label='fix', color='orange', linestyle='--', linewidth=linewith)

    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('f(u)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1,  plot_2],
               labels=['exact', 'trained',],
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures_d/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = name + '_real_prediction_function' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_predict_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/200   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x_label, fix_timestep_predict_data, label='prediction', color='blue', linestyle='--', linewidth=linewith)
    # plt.title(label='X', fontsize=title_fontsize)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2],
               labels=['exact', 'prediction'],
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    save_dir = 'figures/' + name + '/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = name + '_' +'time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_real_and_predict_data(data, example_index, name=''):
    # fig, axes = plt.plots(figsize=(15, 10), sharey=True)
    plt.figure(figsize=(16, 9),dpi=300)
    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50
    cbar_size = 50
    g1 = sns.heatmap(np.flip(data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, annot_kws={"size":30})
    g1.set_ylabel('T', fontsize=label_fontsize)
    g1.set_xlabel('X', fontsize=label_fontsize)
    g1.set_xticklabels([])
    g1.set_yticklabels([])
    g1.set_title("exact u(x, t)", fontsize=title_fontsize)
    # g1.set_title("prediction u(x, t)", fontsize=title_fontsize)
    cax1 = plt.gcf().axes[-1]
    cax1.tick_params(labelsize=cbar_size)
    cax1.spines['top'].set_linewidth(linewith_frame)
    cax1.spines['bottom'].set_linewidth(linewith_frame)
    cax1.spines['right'].set_linewidth(linewith_frame)
    cax1.spines['left'].set_linewidth(linewith_frame)

    save_dir = 'figures/' + name + '/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = 'real_predict_all_time_data_real' + '.pdf'
    plt.savefig(save_dir + file_name)


# def plot_real_and_predict_data(real_data, predict_data, example_index, name=''):
#     fig, axes = plt.subplots(1, 2, figsize=(50, 16), sharey=True)
#     linewith = 10
#     linewith_frame = 4
#     title_fontsize = 60
#     label_fontsize = 50
#     ticks_fontsize = 50
#     cbar_size = 50
#     g1 = sns.heatmap(np.flip(real_data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, annot_kws={"size":30}, ax=axes[0])
#     g1.set_ylabel('T', fontsize=label_fontsize)
#     g1.set_xlabel('X', fontsize=label_fontsize)
#     g1.set_xticklabels([])
#     g1.set_yticklabels([])
#     # plt.xticks(np.arange(0, 2, step=0.2),list('abcdefghigk'),rotation=45)
#     g1.set_title("exact u(x, t)", fontsize=title_fontsize)
#     cax1 = plt.gcf().axes[-1]
#     cax1.tick_params(labelsize=cbar_size)
#     cax1.spines['top'].set_linewidth(linewith_frame)
#     cax1.spines['bottom'].set_linewidth(linewith_frame)
#     cax1.spines['right'].set_linewidth(linewith_frame)
#     cax1.spines['left'].set_linewidth(linewith_frame)
#
#     g2 = sns.heatmap(np.flip(predict_data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, ax=axes[1])
#     g2.set_ylabel('T', fontsize=label_fontsize)
#     g2.set_xlabel('X', fontsize=label_fontsize)
#     g2.set_xticklabels([])
#     g2.set_yticklabels([])
#     g2.set_title("prediction u(x, t)", fontsize=title_fontsize)
#     cax2 = plt.gcf().axes[-1]
#     cax2.tick_params(labelsize=cbar_size)
#     cax2.spines['top'].set_linewidth(linewith_frame)
#     cax2.spines['bottom'].set_linewidth(linewith_frame)
#     cax2.spines['right'].set_linewidth(linewith_frame)
#     cax2.spines['left'].set_linewidth(linewith_frame)
#
#     save_dir = 'figures/'
#     if not os.path.isdir(save_dir):
#         os.mkdir(save_dir)
#     file_name = name + '_real_predict_all_time_data' + '.pdf'
#     fig.savefig(save_dir + file_name)



if __name__ == "__main__":
    # x = []
    # N = 100
    # dx = 1.0/N
    # for i in range(N + 1):
    #     x.append(i*dx)
    # grav = -9
    # power_nb = 6
    # if power_nb == 3:
    #     molecular_coe = [0.0074, -0.3308,  1.1144, -0.1477,  0.0072, -0.3240,  1.0923, -0.0537]
    #     denominator_coe = [0.1489, -0.1299, -0.0212,  0.6736,  0.1567, -0.1505, -0.0136,  0.7577]
    # elif power_nb == 4:
    #     molecular_coe = [-0.0541, -0.0706,  1.1108,  0.7579, -1.0867, -0.0565,  0.0473,  0.1767,
    #                      -0.3883, -0.0614]
    #     denominator_coe = [0.1814,  0.2869,  0.2316,  0.1915,  0.0812,  0.1917, -0.1004, -0.0532,
    #                         0.0145, -0.0766]
    # elif power_nb == 5:
    #     molecular_coe = [3.5134e-04, -5.9416e-02,  8.2195e-01, -2.6555e-02,  2.2711e-01,
    #                       1.6549e+00, -1.2512e-04, -2.1414e-02, -1.7611e-01,  2.6334e-01,
    #                       3.4389e-02,  1.4773e-01]
    #     denominator_coe = [1.7094e-01, -2.9890e-01,  4.5320e-01,  5.2147e-01,  4.1287e-01,
    #                         1.3492e+00, -3.9574e-04,  3.8202e-02, -1.6535e-01,  7.1033e-02,
    #                         1.2314e-01,  1.8307e-01]
    # elif power_nb == 6:
    #     molecular_coe = [-7.0699e-04,  1.0301e-01, -1.2928e+00, -1.0959e+00, -2.8249e-01,
    #                      -3.5936e+00, -7.7790e+00,  1.4742e-04,  3.7239e-02,  2.4613e-01,
    #                      4.2619e-01, -2.4086e+00,  3.4410e+00, -3.0852e+00]
    #     denominator_coe = [-2.6457e-01,  1.3283e-01, -1.2820e-01, -1.7147e+00, -1.6430e+00,
    #                        -2.1634e+00, -8.1376e+00,  2.6369e-03, -9.7678e-02,  4.7533e-01,
    #                        -6.4356e-01,  7.3440e-02,  1.4473e+00, -2.6043e+00]
    # elif power_nb == 7:
    #     molecular_coe = [2.3122e-03, -2.3222e-01,  5.6972e-01,  5.9203e-01,  1.9172e-01,
    #                       -5.7805e-03,  2.4819e-01,  4.0770e-01,  1.4875e-03, -2.2347e-01,
    #                       6.3686e-01,  4.2491e-01,  2.1280e-01,  2.4168e-01,  4.2672e-01,
    #                       1.8061e+00]
    #     denominator_coe = [0.1497, -0.0508, -0.1466,  0.2429,  0.7619,  0.7545,  0.3501, -0.3188,
    #                         0.1725, -0.1388, -0.0269,  0.3828,  0.3654,  0.8305,  0.9993,  0.9177]
    # elif power_nb == 8:
    #     molecular_coe = [0.0297, -0.2006, -0.2684,  0.0892,  0.2973, -0.1894, -0.7679, -0.8349,
    #                       0.7015,  0.0093,  0.1088, -0.5055, -0.1162,  0.6518,  0.8877,  0.6654,
    #                       -0.1231, -0.7964]
    #     denominator_coe = [-0.1775,  0.1747, -0.2943, -0.0108, -0.0834, -0.2282, -0.1190, -0.1271,
    #                        -0.3280, -0.0847, -0.0100,  0.1639,  0.1523,  0.1541,  0.0890,  0.1646,
    #                        0.1449,  0.0945]
    # y_gt = np.array([f_gt(ele, grav) for ele in x])
    # y_pd = np.array([f_pf(ele, grav, molecular_coe, denominator_coe, power_nb) for ele in x])
    # name = 'test_grav_{beta}_power_{power}'.format(beta=grav, power=power_nb)
    # plot_real_and_predict_function(x, y_gt, y_pd, name=name)

    # # # 画出不同时刻的预测和真实的解
    beta_list = [-3, -1, 9]
    for power_nb in [3, 4, 5, 6, 7, 8]:
        for beta in beta_list:
            if beta < 0:
                beta_name = 'ne_' + str(np.abs(beta))
            else:
                beta_name = str(beta)
            experiment_name = 'N_400_example_4_beta_' + beta_name + '_power_np_' + str(power_nb)
            data_dir = 'data_m/' + experiment_name + '/'
            real_data_file_1 = '/' + experiment_name + '_real_U' + '.npy'
            predict_data_file_1 = '/' + experiment_name + '_predict_U' + '.npy'
            real_data_file = data_dir + real_data_file_1
            predict_data_file = data_dir + predict_data_file_1
            real_data = np.load(real_data_file)
            predict_data = np.load(predict_data_file)
            # power_nb == 3
            if power_nb == 3 and beta == -5:
                dt_real = 0.003984
                dt_predict = 0.003984
            elif power_nb == 3 and beta == -3:
                dt_real = 0.004866
                dt_predict = 0.004175
            elif power_nb == 3 and beta == -1:
                dt_real = 0.005848
                dt_predict = 0.006920
            elif power_nb == 3 and beta == 9:
                dt_real = 0.003540
                dt_predict = 0.003021
            # power_nb == 4
            elif power_nb == 4 and beta == -5:
                dt_real = 0.003984
                dt_predict = 0.003984
            elif power_nb == 4 and beta == -3:
                dt_real = 0.004866
                dt_predict = 0.006192
            elif power_nb == 4 and beta == -1:
                dt_real = 0.005848
                dt_predict = 0.006969
            elif power_nb == 4 and beta == 9:
                dt_real = 0.003540
                dt_predict = 0.002096
            # power_nb == 5
            elif power_nb == 5 and beta == -5:
                dt_real = 0.003984
                dt_predict = 0.003984
            elif power_nb == 5 and beta == -3:
                dt_real = 0.004866
                dt_predict = 0.006116
            elif power_nb == 5 and beta == -1:
                dt_real = 0.005848
                dt_predict = 0.007246
            elif power_nb == 5 and beta == 9:
                dt_real = 0.003540
                dt_predict = 0.004082
            # power_nb == 6
            elif power_nb == 6 and beta == -5:
                dt_real = 0.003984
                dt_predict = 0.005141
            elif power_nb == 6 and beta == -3:
                dt_real = 0.004866
                dt_predict = 0.005952
            elif power_nb == 6 and beta == -1:
                dt_real = 0.005848
                dt_predict = 0.007092
            elif power_nb == 6 and beta == 9:
                dt_real = 0.003540
                dt_predict = 0.003953
            # power_nb == 7
            elif power_nb == 7 and beta == -5:
                dt_real = 0.003984
                dt_predict = 0.003984
            elif power_nb == 7 and beta == -3:
                dt_real = 0.004866
                dt_predict = 0.005747
            elif power_nb == 7 and beta == -1:
                dt_real = 0.005848
                dt_predict = 0.006757
            elif power_nb == 7 and beta == 9:
                dt_real = 0.003540
                dt_predict = 0.004283
            # power_nb == 8
            elif power_nb == 8 and beta == -5:
                dt_real = 0.003984
                dt_predict = 0.004000
            elif power_nb == 8 and beta == -3:
                dt_real = 0.004866
                dt_predict = 0.005634
            elif power_nb == 8 and beta == -1:
                dt_real = 0.005848
                dt_predict = 0.006623
            elif power_nb == 8 and beta == 9:
                dt_real = 0.003540
                dt_predict = 0.004141

            obs_t = []
            for i in range(1, 10):
                obs_t.append(0.1 * i)
            real_time_step = []
            for ele in obs_t:
                real_time_step.append(round(ele/dt_real))
            predict_time_step = []
            for ele in obs_t:
                predict_time_step.append(round(ele/dt_predict))
            # 计算出 0.1-0.9 的mse
            print("\033[34mbeta %.6f, power_nb %.6f,\033[0m" % (beta, power_nb))
            print(mse(torch.from_numpy(real_data[real_time_step, :, :]), torch.from_numpy(predict_data[predict_time_step, :, :])))

            # # 画出0.3(2)， 0.6(5)， 0.9(8) 时刻的解得图示
            # for i in [2, 5, 8]:
            #     time_steps_real = real_time_step[i]
            #     time_steps_predict = predict_time_step[i]
            #     if i == 2:
            #         time = 0.3
            #     elif i == 5:
            #         time = 0.6
            #     elif i == 8:
            #         time = 0.9
            #     example_index = 0
            #     cell_numbers = 400
            #     test_data_name = "N_400_example_4_beta_" + beta_name + "_power_np_" + str(power_nb)
            #     plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, name=test_data_name)


    # power_nb = 6
    # beta_name = 'ne_7'
    # example_index = 0
    # experiment_name = 'N_400_example_4_beta_' + beta_name + '_power_np_' + str(power_nb)
    # data_dir = 'data/' + experiment_name + '/'
    # real_data_file_1 = '/' + experiment_name + '_real_U' + '.npy'
    # predict_data_file_1 = '/' + experiment_name + '_predict_U' + '.npy'
    # real_data_file = data_dir + real_data_file_1
    # predict_data_file = data_dir + predict_data_file_1
    # real_data = np.load(real_data_file)
    # predict_data = np.load(predict_data_file)
    # test_data_name = experiment_name
    # plot_real_and_predict_data(real_data, example_index, name=test_data_name)



