import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import re
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
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
    # coe = [8.4920e-03,  1.3823e+00,  1.9598e-01, -3.3295e-01, -3.9308e-01,
    #   -1.8213e-02,  1.5651e-01, -1.7243e-05,  2.0499e-03,  4.8243e-02,
    #   -1.2439e-01,  5.9704e-02,  5.1649e-02, -3.7291e-02]
    # 5
    # coe = [7.3179e-03,  1.4015e+00,  1.6697e-01, -4.3744e-01, -2.3990e-01,
    #   9.6361e-02, -2.6094e-04,  8.7118e-03,  7.7761e-03, -4.1227e-02,
    #   1.8892e-02,  6.3432e-03]
    # 4
    # coe = [ 8.7882e-03,  1.4095e+00,  5.6999e-02, -2.5414e-01, -2.3271e-01,
    #   -1.8903e-04,  6.8922e-03,  1.8202e-02, -6.3637e-02,  3.9014e-02]
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

    # 画出损失函数的效率
    name = "loss"
    list_loss = []
    with open("beta_200/checkpoint/N_400_example_6_dt_0.1_layer_10_beta_200/loss.txt", encoding="utf-8") as f:
        for line in f.readlines():
            if line.startswith("loss0"):
                list_loss.append(float(line.split(',')[0].split()[-1]))

    # list_loss.remove(55.283942)
    x = np.arange(0, len(list_loss))
    linewith = 5
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    plot_1, = plt.plot(x, list_loss, label='loss', color='red', linestyle='-', linewidth=linewith)
    plt.xlabel('Iterations', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')

    # 添加第一个子图
    label_size = 25
    linewith_frame = 3
    linewith = 3.0
    axins1 = ax1.inset_axes((0.08, 0.1, 0.90, 0.75))
    axins1.plot(x, list_loss, color='red', linestyle='-', linewidth=linewith)
    # # 设置放大区间
    zone_left1 = 2
    zone_right1 = 206
    # X轴的显示范围
    xlim10 = x[zone_left1]
    xlim11 = x[zone_right1]
    # Y轴的显示范围
    y1 = np.hstack((list_loss[zone_left1:zone_right1]))
    ylim10 = np.min(y1)
    ylim11 = np.max(y1)
    # 调整子坐标系的显示范围
    axins1.set_xlim(xlim10, xlim11)
    axins1.set_ylim(ylim10, ylim11)
    axins1.tick_params(axis="x", labelsize=label_size)
    axins1.tick_params(axis="y", labelsize=label_size)
    axins1.spines['top'].set_linewidth(linewith_frame)
    axins1.spines['bottom'].set_linewidth(linewith_frame)
    axins1.spines['right'].set_linewidth(linewith_frame)
    axins1.spines['left'].set_linewidth(linewith_frame)
    mark_inset(ax1, axins1, loc1=3, loc2=4, fc="none", ec='blue', lw=1.5, alpha=0.8, ls='--')
    plt.show()

    save_dir = 'figures/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = name + '/beta_300_poly_loss' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

    # poly loss
    # name = 'loss'
    # list_loss_poly = []
    # with open("./beta_300_poly/checkpoint/N_400_example_9_dt_0.1_layer_10_beta_300_poly_1/loss.txt", encoding="utf-8") as f:
    #     for line in f.readlines():
    #         if line.startswith("loss0"):
    #             list_loss_poly.append(float(line.split(',')[0].split()[-1]))
    # list_loss_poly.insert(0, 60.0345)
    # x = np.arange(0, len(list_loss_poly))
    # linewith = 5
    # linewith_frame = 4
    # title_fontsize = 60
    # label_fontsize = 50
    # ticks_fontsize = 50
    #
    # fig = plt.figure(figsize=(40, 20))
    # plot_1, = plt.plot(x, list_loss_poly, label='loss', color='red', linestyle='-', linewidth=linewith)
    # plt.xlabel('Iterations', fontsize=label_fontsize)
    # plt.ylabel('Loss', fontsize=label_fontsize)
    # plt.xticks(fontsize=ticks_fontsize)
    # plt.yticks(fontsize=ticks_fontsize)
    # ax1 = plt.gca()
    # ax1.spines['top'].set_linewidth(linewith_frame)
    # ax1.spines['bottom'].set_linewidth(linewith_frame)
    # ax1.spines['right'].set_linewidth(linewith_frame)
    # ax1.spines['left'].set_linewidth(linewith_frame)
    # plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    #
    # # 添加第一个子图
    # label_size = 25
    # linewith_frame = 3
    # linewith = 3.0
    # axins1 = ax1.inset_axes((0.08, 0.1, 0.90, 0.75))
    # axins1.plot(x, list_loss_poly, color='red', linestyle='-', linewidth=linewith)
    # # # 设置放大区间
    # zone_left1 = 2
    # zone_right1 = 118
    # # X轴的显示范围
    # xlim10 = x[zone_left1]
    # xlim11 = x[zone_right1]
    # # Y轴的显示范围
    # y1 = np.hstack((list_loss_poly[zone_left1:zone_right1]))
    # ylim10 = np.min(y1)
    # ylim11 = np.max(y1)
    # # 调整子坐标系的显示范围
    # axins1.set_xlim(xlim10, xlim11)
    # axins1.set_ylim(ylim10, ylim11)
    # axins1.tick_params(axis="x", labelsize=label_size)
    # axins1.tick_params(axis="y", labelsize=label_size)
    # axins1.spines['top'].set_linewidth(linewith_frame)
    # axins1.spines['bottom'].set_linewidth(linewith_frame)
    # axins1.spines['right'].set_linewidth(linewith_frame)
    # axins1.spines['left'].set_linewidth(linewith_frame)
    # mark_inset(ax1, axins1, loc1=3, loc2=4, fc="none", ec='blue', lw=1.5, alpha=0.8, ls='--')
    # plt.show()
    #
    # save_dir = 'figures/'
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # file_name = name + '/beta_300_loss' + '.pdf'
    # fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)
    #




    # 画出不同时刻的预测和真实的解
    # beta_list = [-150, -100, -50, 350, 400]
    # for power_nb in [8]:
    #     for beta in beta_list:
    #         if beta < 0:
    #             beta_name = 'ne_' + str(np.abs(beta))
    #         else:
    #             beta_name = str(beta)
    #         experiment_name = 'N_400_example_4_beta_' + beta_name + '_power_np_' + str(power_nb)
    #         data_dir = 'data/' + experiment_name + '/'
    #         real_data_file_1 = '/' + experiment_name + '_real_U' + '.npy'
    #         predict_data_file_1 = '/' + experiment_name + '_predict_U' + '.npy'
    #         real_data_file = data_dir + real_data_file_1
    #         predict_data_file = data_dir + predict_data_file_1
    #         real_data = np.load(real_data_file)
    #         predict_data = np.load(predict_data_file)
    #         if power_nb == 1 and beta == -150:
    #             dt_real = 0.009479
    #             dt_predict = 0.015504
    #         elif power_nb == 1 and beta == -100:
    #             dt_real = 0.011429
    #             dt_predict = 0.016260
    #         elif power_nb == 1 and beta == -50:
    #             dt_real = 0.012422
    #             dt_predict = 0.016949
    #         elif power_nb == 1 and beta == 350:
    #             dt_real = 0.003984
    #             dt_predict = 0.025974
    #         elif power_nb == 1 and beta == 400:
    #             dt_real = 0.003630
    #             dt_predict = 0.027778
    #         # power_nb == 2
    #         elif power_nb == 2 and beta == -150:
    #             dt_real = 0.009479
    #             dt_predict = 0.009302
    #         elif power_nb == 2 and beta == -100:
    #             dt_real = 0.011429
    #             dt_predict = 0.012903
    #         elif power_nb == 2 and beta == -50:
    #             dt_real = 0.012422
    #             dt_predict = 0.014085
    #         elif power_nb == 2 and beta == 350:
    #             dt_real = 0.003984
    #             dt_predict = 0.003711
    #         elif power_nb == 2 and beta == 400:
    #             dt_real = 0.003630
    #             dt_predict = 0.003396
    #         # power_nb == 3
    #         elif power_nb == 3 and beta == -150:
    #             dt_real = 0.009479
    #             dt_predict = 0.010989
    #         elif power_nb == 3 and beta == -100:
    #             dt_real = 0.011429
    #             dt_predict = 0.012579
    #         elif power_nb == 3 and beta == -50:
    #             dt_real = 0.012422
    #             dt_predict = 0.014085
    #         elif power_nb == 3 and beta == 350:
    #             dt_real = 0.003984
    #             dt_predict = 0.002454
    #         elif power_nb == 3 and beta == 400:
    #             dt_real = 0.003630
    #             dt_predict = 0.002198
    #         # power_nb == 4
    #         elif power_nb == 4 and beta == -150:
    #             dt_real = 0.009479
    #             dt_predict = 0.009346
    #         elif power_nb == 4 and beta == -100:
    #             dt_real = 0.011429
    #             dt_predict = 0.011364
    #         elif power_nb == 4 and beta == -50:
    #             dt_real = 0.012422
    #             dt_predict = 0.013986
    #         elif power_nb == 4 and beta == 350:
    #             dt_real = 0.003984
    #             dt_predict = 0.004167
    #         elif power_nb == 4 and beta == 400:
    #             dt_real = 0.003630
    #             dt_predict = 0.003795
    #         # power_nb == 5
    #         elif power_nb == 5 and beta == -150:
    #             dt_real = 0.009479
    #             dt_predict = 0.009615
    #         elif power_nb == 5 and beta == -100:
    #             dt_real = 0.011429
    #             dt_predict = 0.011696
    #         elif power_nb == 5 and beta == -50:
    #             dt_real = 0.012422
    #             dt_predict = 0.014286
    #         elif power_nb == 5 and beta == 350:
    #             dt_real = 0.003984
    #             dt_predict = 0.004024
    #         elif power_nb == 5 and beta == 400:
    #             dt_real = 0.003630
    #             dt_predict = 0.003663
    #         # power_nb == 6
    #         elif power_nb == 6 and beta == -150:
    #             dt_real = 0.009479
    #             dt_predict = 0.009346
    #         elif power_nb == 6 and beta == -100:
    #             dt_real = 0.011429
    #             dt_predict = 0.011236
    #         elif power_nb == 6 and beta == -50:
    #             dt_real = 0.012422
    #             dt_predict = 0.013699
    #         elif power_nb == 6 and beta == 350:
    #             dt_real = 0.003984
    #             dt_predict = 0.004082
    #         elif power_nb == 6 and beta == 400:
    #             dt_real = 0.003630
    #             dt_predict = 0.003717
    #         # power_nb == 7
    #         elif power_nb == 7 and beta == -150:
    #             dt_real = 0.00947
    #             dt_predict = 0.009132
    #         elif power_nb == 7 and beta == -100:
    #             dt_real = 0.011429
    #             dt_predict = 0.010989
    #         elif power_nb == 7 and beta == -50:
    #             dt_real = 0.012422
    #             dt_predict = 0.013514
    #         elif power_nb == 7 and beta == 350:
    #             dt_real = 0.003984
    #             dt_predict = 0.004175
    #         elif power_nb == 7 and beta == 400:
    #             dt_real = 0.003630
    #             dt_predict = 0.003802
    #         # power_nb == 8
    #         elif power_nb == 8 and beta == -150:
    #             dt_real = 0.009479
    #             dt_predict = 0.008850
    #         elif power_nb == 8 and beta == -100:
    #             dt_real = 0.011429
    #             dt_predict = 0.010753
    #         elif power_nb == 8 and beta == -50:
    #             dt_real = 0.012422
    #             dt_predict = 0.013333
    #         elif power_nb == 8 and beta == 350:
    #             dt_real = 0.003984
    #             dt_predict = 0.004367
    #         elif power_nb == 8 and beta == 400:
    #             dt_real = 0.003630
    #             dt_predict = 0.003984
    #         obs_t = []
    #         for i in range(1, 10):
    #             obs_t.append(0.1 * i)
    #         real_time_step = []
    #         for ele in obs_t:
    #             real_time_step.append(round(ele/dt_real))
    #         predict_time_step = []
    #         for ele in obs_t:
    #             predict_time_step.append(round(ele/dt_predict))
    #         # print(real_time_step)
    #         # print(predict_time_step)
    #         # 计算出 0.1-0.9 的mse
    #         print("\033[34mbeta %.6f, power_nb %.6f,\033[0m" % (beta, power_nb))
    #         print(mse(torch.from_numpy(real_data[real_time_step, :, :]), torch.from_numpy(predict_data[predict_time_step, :, :])))
    #
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
