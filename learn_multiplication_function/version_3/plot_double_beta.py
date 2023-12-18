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
def f_gt(u, beta):
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * beta * np.power(u, 2) * ((3/4) - 2*u +(3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f

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
    save_dir = 'figures/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_real_prediction_function' + '.pdf'
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
    # plt.show()

    save_dir = 'figures/' + name + '/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = name + '_' +'time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_mul_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_predict_data = predict_data[time_steps_predict, example_index, :]

    linewith = 3
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/200   # 0.05
    for i in range(fix_timestep_real_data.shape[1]):
        x_label.append(i * dx)
    # t = 0.3
    plot_1, = plt.plot(x_label, fix_timestep_real_data[0], label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x_label, fix_timestep_predict_data[0], label='prediction', color='blue', linestyle='--', linewidth=linewith)
    # t = 0.6
    plot_3, = plt.plot(x_label, fix_timestep_real_data[1], label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_4, = plt.plot(x_label, fix_timestep_predict_data[1], label='prediction', color='blue', linestyle='--', linewidth=linewith)
    # t = 0.9.
    plot_5, = plt.plot(x_label, fix_timestep_real_data[2], label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_6, = plt.plot(x_label, fix_timestep_predict_data[2], label='prediction', color='blue', linestyle='--', linewidth=linewith)

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
    #
    # save_dir = 'figures/' + name + '/'
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # file_name = name + '_' +'time_' + str(time) + '.pdf'
    # fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

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
#     save_dir = 'figures/' + name + '/'
#     if not os.path.isdir(save_dir):
#         os.mkdir(save_dir)
#     file_name = 'real_predict_all_time_data' + '.pdf'
#     fig.savefig(save_dir + file_name)





if __name__ == "__main__":
    # # 得出系数
    # 6
    # coe = [8.4914e-02,  9.9527e-01,  3.0034e-01,  4.8697e-03,  1.9316e-01,
    #        -2.4122e-01, -3.9025e-01, -7.9987e-04,  1.7935e-02, -1.6340e-02,
    #        -8.0825e-02,  1.6248e-01, -7.7874e-02, -6.3481e-03, -1.6276e-05,
    #        6.8954e-04, -6.9992e-03,  2.8141e-02, -5.2477e-02,  4.5597e-02,
    #        -1.4941e-02]
    # 5
    # coe = [2.9751e-01,  3.3672e-01,  3.9370e-01,  7.2820e-02,  4.3036e-02,
    #           5.3018e-02, -7.9972e-03,  5.6774e-02, -4.8435e-02, -2.9942e-02,
    #          -2.9883e-02,  6.1479e-02,  3.8466e-05, -7.3222e-04,  4.0131e-03,
    #          -9.9575e-03,  1.1080e-02, -4.4993e-03]
    # 4
    # coe = [1.8309e-01,  7.8826e-01,  8.8490e-02,  1.5428e-01, -8.0013e-02,
    #   2.8330e-03, -2.9903e-02,  7.4825e-02, -1.9983e-02, -3.3688e-02,
    #  -1.3798e-05,  1.4567e-04, -2.0892e-04, -1.7257e-04,  2.7053e-04]
    # 3
    coe = [2.2105e-01,  5.6923e-01,  3.3248e-01,  5.2441e-02, -3.3481e-03,
           2.9955e-02, -4.0769e-02,  1.2185e-02,  6.6561e-06, -3.3181e-05,
           3.3420e-05, -3.2478e-06]
    # 测试新的beta
    # x = []
    # N = 100
    # dx = 1.0/N
    # for i in range(N + 1):
    #     x.append(i*dx)
    # beta = 500
    # power_nb = 6
    # y_gt = np.array([f_gt(ele, beta) for ele in x])
    # y_pd = np.array([f_pf(ele, beta, coe, power_nb) for ele in x])
    # name = "test_beta_500_power_nb_6"
    # plot_real_and_predict_function(x, y_gt, y_pd, name=name)

    # 画出不同时刻的预测和真实的解
    # beta_list = [-150, -100, -50, 350, 400]
    beta_list = [-50, 350]
    for power_nb in [1, 2, 3, 4]:
        for beta in beta_list:
            if beta < 0:
                beta_name = 'ne_' + str(np.abs(beta))
            else:
                beta_name = str(beta)
            experiment_name = 'N_400_example_4_beta_' + beta_name + '_power_np_' + str(power_nb)
            data_dir = 'data_double_beta/' + experiment_name + '/'
            real_data_file_1 = '/' + experiment_name + '_real_U' + '.npy'
            predict_data_file_1 = '/' + experiment_name + '_predict_U' + '.npy'
            real_data_file = data_dir + real_data_file_1
            predict_data_file = data_dir + predict_data_file_1
            real_data = np.load(real_data_file)
            predict_data = np.load(predict_data_file)
            # power_nb == 1
            if power_nb == 1 and beta == -50:
                dt_real = 0.012422
                dt_predict = 0.017699
            elif power_nb == 1 and beta == 350:
                dt_real = 0.003984
                dt_predict = 0.026316
            # power_nb == 2
            elif power_nb == 2 and beta == -50:
                dt_real = 0.012422
                dt_predict = 0.011765
            elif power_nb == 2 and beta == 350:
                dt_real = 0.003984
                dt_predict = 0.003617
            # power_nb == 3
            elif power_nb == 3 and beta == -50:
                dt_real = 0.012422
                dt_predict = 0.008097
            elif power_nb == 3 and beta == 350:
                dt_real = 0.003984
                dt_predict = 0.001579
            # power_nb == 4
            elif power_nb == 4 and beta == -50:
                dt_real = 0.012422
                dt_predict = 0.003350
            elif power_nb == 4 and beta == 350:
                dt_real = 0.003984
                dt_predict = 0.001754
            # # power_nb == 5
            # elif power_nb == 5 and beta == -50:
            #     dt_real = 0.012422
            #     dt_predict = 0.004320
            # elif power_nb == 5 and beta == 350:
            #     dt_real = 0.003984
            #     dt_predict = 0.000266
            # # power_nb == 6
            # elif power_nb == 6 and beta == -50:
            #     dt_real = 0.012422
            #     dt_predict =
            # elif power_nb == 6 and beta == 350:
            #     dt_real = 0.003984
            #     dt_predict =
            obs_t = []
            for i in range(1, 10):
                obs_t.append(0.1 * i)
            real_time_step = []
            for ele in obs_t:
                real_time_step.append(round(ele/dt_real))
            predict_time_step = []
            for ele in obs_t:
                predict_time_step.append(round(ele/dt_predict))
            # print(real_time_step)
            # print(predict_time_step)
            # 计算出 0.1-0.9 的mse
            print("\033[34mbeta %.6f, power_nb %.6f,\033[0m" % (beta, power_nb))
            print(mse(torch.from_numpy(real_data[real_time_step, :, :]), torch.from_numpy(predict_data[predict_time_step, :, :])))

    #         # # 画出0.3(2)， 0.6(5)， 0.9(8) 时刻的解得图示
    #         for i in [2, 5, 8]:
    #             time_steps_real = real_time_step[i]
    #             time_steps_predict = predict_time_step[i]
    #             if i == 2:
    #                 time = 0.3
    #             elif i == 5:
    #                 time = 0.6
    #             elif i == 8:
    #                 time = 0.9
    #             example_index = 0
    #             cell_numbers = 400
    #             test_data_name = "N_400_example_4_beta_" + beta_name + "_power_np_" + str(power_nb)
    #             plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, name=test_data_name)
    #
    # power_nb = 6
    # beta_name = '400'
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
