import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from scipy.integrate import solve_ivp
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import random
np.random.seed(0)
torch.manual_seed(0)


class Data:
    def __init__(self, power_nb, beta_list):
        self.x = []
        N = 400
        dx = 1.0/N
        for i in range(N + 1):
            self.x.append(i*dx)
        self.start = 0
        self.end = 401
        self.power_nb = power_nb
        self.beta_list = beta_list

    # 给出第一步骤求解出来的表达式
    #  beta = 10
    def pd_f_beta_10(self, u):
        # f = (1.560993436326928)*u+(-0.9634805041085234)*u**3+(0.4764576323582725)*1+(0.31051547278263564)*u**4+(0.11337087374179529)*u**2+(-0.021331490250448778)*u**5+(0.0005608257887212879)*u**6+(-5.965207721685407e-06)*u**7
        molecular = (1.7006642479391998)*u+(-1.6452598353864365)*1+(-0.9640845280843208)*u**3+(0.32141408878787053)*u**4+(-0.019774328349539017)*u**2+(-0.0035837572880650964)*u**6+(-0.0029507141087210752)*u**5+(0.0029023054632591077)*u**7+(-0.00048249102443053484)*u**8
        denominator = (1.0205228295019033)*1+(-0.06627542485672068)*u+(0.043123192645832815)*u**2+(0.002539329424290226)*u**3+(0.0017753486411678139)*u**6+(0.0014617469494484089)*u**5+(-0.0014377659105460562)*u**7+(-0.001274020362540719)*u**4+(0.00023902003281614455)*u**8
        return molecular/denominator

    # beta = 130
    def pd_f_beta_130(self, u):
        molecular = (-1.2968745891186226)*u**2+(1.2216010064291682)*u+(-0.9677141953929043)*u**3+(0.6587989187866976)*u**4+(-0.5409501142739394)*1+(0.16876825243351062)*u**5+(-0.05534748569220814)*u**6+(-0.045059687236713324)*u**7+(0.01711133819504112)*u**8
        denominator = (0.985669509243963)*u**2+(0.5975083501548867)*u**3+(-0.4796509733311272)*u**4+(-0.37909216023769804)*u+(0.3038064060440057)*1+(0.05726020972853409)*u**5+(-0.018778464509676938)*u**6+(-0.015287988731724553)*u**7+(0.005805587245562908)*u**8
        return molecular/denominator

    # beta = 200
    def pd_f_beta_200(self, u):
        molecular = (-2.260851548587051)*u**2+(1.7718349839154381)*u+(-0.6949712732370981)*1+(-0.3600454113745229)*u**3+(-0.0181190248537432)*u**4+(-0.014385316652640883)*u**5+(-0.00046453569129470634)*u**6+(0.00011901151409314868)*u**7+(-3.004024713501341e-06)*u**8
        denominator = (0.7180772019500903)*u**2+(-0.40927278015013746)*u+(0.2220316882312772)*1+(0.11192666278786897)*u**4+(0.04638811040704861)*u**3+(0.045289444878963955)*u**5+(0.0014625026402418971)*u**6+(-0.0003746852111520395)*u**7+(9.457602843395738e-06)*u**8
        return molecular/denominator

    # beta = 300
    def pd_f_beta_300(self, u):
        # f = (90.22074167375786)*u**4+(-75.93382190422577)*u**3+(-52.336379628508816)*u**5+(24.51097480913004)*u**2+(15.799670993686469)*u**6+(-2.3843711960314695)*u**7+(0.9694337282559484)*u+(0.4319751725853378)*1+(0.14210058031888906)*u**8
        molecular = (-3.184890516872588)*u**2+(2.599943994084381)*u+(-0.9848341667546185)*1+(-0.28636677088182)*u**4+(-0.15679871345469737)*u**3+(-0.09813917149173579)*u**5+(-0.007063615029956347)*u**6+(2.001463803709729e-05)*u**7
        denominator = (0.6433489418390006)*u**2+(-0.42933982164898316)*u+(0.20826799467997512)*1+(0.1709331010002923)*u**4+(-0.0886916474469701)*u**3+(0.058492775069557204)*u**5+(0.004210046190984737)*u**6+(-1.1929097250440108e-05)*u**7
        return molecular/denominator

    # 根据之前的case, 分别计算f(u)在不同u处的值
    def buid_target_of_beta(self, beta):
        if beta == 10:
            d = np.array([self.pd_f_beta_10(ele) - self.pd_f_beta_10(0) for ele in self.x])
        elif beta == 130:
            d = np.array([self.pd_f_beta_130(ele) - self.pd_f_beta_130(0) for ele in self.x])
        elif beta == 200:
            d = np.array([self.pd_f_beta_200(ele) - self.pd_f_beta_200(0) for ele in self.x])
        elif beta == 300:
            d = np.array([self.pd_f_beta_300(ele) - self.pd_f_beta_300(0) for ele in self.x])
        return d

    def build_targets(self):
        res = []
        for beta in self.beta_list:
            d = self.buid_target_of_beta(beta)
            if len(res) == 0:
                res = d[self.start:self.end, np.newaxis]
            else:
                res = np.vstack((res, d[self.start:self.end, np.newaxis]))
        return res

    # 构建特征x
    # 假设函数的表示式：f(u)=a0 + a1*u + a2*u**2 + a3*u**3 + a4*u**4 + a5*u**5 + a6*u**6 + beta *(b0 + b1*u + b2*u**2 + b3* u**3 + b4*u**4 + b5*u**5 + b6*u**6)
    # 接下来打算学习a1, a2, ....这个参数， 对应的特征就是1, u, u**2,...
    # 不包含beta的项
    def v_0(self, u):
        v = 1
        return v
    def v_1(self, u):
        v = u
        return v
    def v_2(self, u):
        v = u**2
        return v
    def v_3(self, u):
        v = u**3
        return v
    def v_4(self, u):
        v = u**4
        return v
    def v_5(self, u):
        v = u**5
        return v
    def v_6(self, u):
        v = u**6
        return v
    def v_7(self, u):
        v = u**7
        return v
    def v_8(self, u):
        v = u**8
        return v

    # 包含beta的项
    def v_beta_0(self, u, beta):
        v = beta
        return v
    def v_beta_1(self, u, beta):
        v = beta * u
        return v
    def v_beta_2(self, u, beta):
        v = beta * u**2
        return v
    def v_beta_3(self, u, beta):
        v = beta * u**3
        return v
    def v_beta_4(self, u, beta):
        v = beta * u**4
        return v
    def v_beta_5(self, u, beta):
        v = beta * u**5
        return v
    def v_beta_6(self, u, beta):
        v = beta * u**6
        return v
    def v_beta_7(self, u, beta):
        v = beta * u**7
        return v
    def v_beta_8(self, u, beta):
        v = beta * u**8
        return v


    # 包含beta^2
    def v_2_beta_0(self, u, beta):
        v = beta**2
        return v
    def v_2_beta_1(self, u, beta):
        v = beta**2 * u
        return v
    def v_2_beta_2(self, u, beta):
        v = beta**2 * u**2
        return v
    def v_2_beta_3(self, u, beta):
        v = beta**2 * u**3
        return v
    def v_2_beta_4(self, u, beta):
        v = beta**2 * u**4
        return v
    def v_2_beta_5(self, u, beta):
        v = beta**2 * u**5
        return v
    def v_2_beta_6(self, u, beta):
        v = beta**2 * u**6
        return v
    def v_2_beta_7(self, u, beta):
        v = beta**2 * u**7
        return v
    def v_2_beta_8(self, u, beta):
        v = beta**2 * u**8
        return v

    def build_feature_of_beta(self, beta):
        v_0 = np.array([self.v_0(ele) for ele in self.x])
        v_1 = np.array([self.v_1(ele) for ele in self.x])
        v_2 = np.array([self.v_2(ele) for ele in self.x])
        v_3 = np.array([self.v_3(ele) for ele in self.x])
        v_4 = np.array([self.v_4(ele) for ele in self.x])
        v_5 = np.array([self.v_5(ele) for ele in self.x])
        v_6 = np.array([self.v_6(ele) for ele in self.x])
        v_7 = np.array([self.v_7(ele) for ele in self.x])
        v_8 = np.array([self.v_8(ele) for ele in self.x])

        v_0_beta = np.array([self.v_beta_0(ele, beta) for ele in self.x])
        v_1_beta = np.array([self.v_beta_1(ele, beta) for ele in self.x])
        v_2_beta = np.array([self.v_beta_2(ele, beta) for ele in self.x])
        v_3_beta = np.array([self.v_beta_3(ele, beta) for ele in self.x])
        v_4_beta = np.array([self.v_beta_4(ele, beta) for ele in self.x])
        v_5_beta = np.array([self.v_beta_5(ele, beta) for ele in self.x])
        v_6_beta = np.array([self.v_beta_6(ele, beta) for ele in self.x])
        v_7_beta = np.array([self.v_beta_7(ele, beta) for ele in self.x])
        v_8_beta = np.array([self.v_beta_8(ele, beta) for ele in self.x])

        v_0_beta_2 = np.array([self.v_2_beta_0(ele, beta) for ele in self.x])
        v_1_beta_2 = np.array([self.v_2_beta_1(ele, beta) for ele in self.x])
        v_2_beta_2 = np.array([self.v_2_beta_2(ele, beta) for ele in self.x])
        v_3_beta_2 = np.array([self.v_2_beta_3(ele, beta) for ele in self.x])
        v_4_beta_2 = np.array([self.v_2_beta_4(ele, beta) for ele in self.x])
        v_5_beta_2 = np.array([self.v_2_beta_5(ele, beta) for ele in self.x])
        v_6_beta_2 = np.array([self.v_2_beta_6(ele, beta) for ele in self.x])
        v_7_beta_2 = np.array([self.v_2_beta_7(ele, beta) for ele in self.x])
        v_8_beta_2 = np.array([self.v_2_beta_8(ele, beta) for ele in self.x])

        if self.power_nb == 1:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis]
                                , v_0_beta_2[self.start:self.end, np.newaxis], v_1_beta_2[self.start:self.end, np.newaxis]))
        elif self.power_nb == 2:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis]
                                , v_0_beta_2[self.start:self.end, np.newaxis], v_1_beta_2[self.start:self.end, np.newaxis], v_2_beta_2[self.start:self.end, np.newaxis]))
        elif self.power_nb == 3:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis]
                                , v_0_beta_2[self.start:self.end, np.newaxis], v_1_beta_2[self.start:self.end, np.newaxis], v_2_beta_2[self.start:self.end, np.newaxis], v_3_beta_2[self.start:self.end, np.newaxis]))
        elif self.power_nb == 4:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis]
                                , v_0_beta_2[self.start:self.end, np.newaxis], v_1_beta_2[self.start:self.end, np.newaxis], v_2_beta_2[self.start:self.end, np.newaxis], v_3_beta_2[self.start:self.end, np.newaxis], v_4_beta_2[self.start:self.end, np.newaxis]))
        elif self.power_nb == 5:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis], v_5[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis], v_5_beta[self.start:self.end, np.newaxis]
                                , v_0_beta_2[self.start:self.end, np.newaxis], v_1_beta_2[self.start:self.end, np.newaxis], v_2_beta_2[self.start:self.end, np.newaxis], v_3_beta_2[self.start:self.end, np.newaxis], v_4_beta_2[self.start:self.end, np.newaxis], v_5_beta_2[self.start:self.end, np.newaxis]))
        elif self.power_nb == 6:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis], v_5[self.start:self.end, np.newaxis], v_6[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis], v_5_beta[self.start:self.end, np.newaxis], v_6_beta[self.start:self.end, np.newaxis]
                                , v_0_beta_2[self.start:self.end, np.newaxis], v_1_beta_2[self.start:self.end, np.newaxis], v_2_beta_2[self.start:self.end, np.newaxis], v_3_beta_2[self.start:self.end, np.newaxis], v_4_beta_2[self.start:self.end, np.newaxis], v_5_beta_2[self.start:self.end, np.newaxis], v_6_beta_2[self.start:self.end, np.newaxis]))
        elif self.power_nb == 7:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis], v_5[self.start:self.end, np.newaxis], v_6[self.start:self.end, np.newaxis], v_7[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis], v_5_beta[self.start:self.end, np.newaxis], v_6_beta[self.start:self.end, np.newaxis], v_7_beta[self.start:self.end, np.newaxis]
                                , v_0_beta_2[self.start:self.end, np.newaxis], v_1_beta_2[self.start:self.end, np.newaxis], v_2_beta_2[self.start:self.end, np.newaxis], v_3_beta_2[self.start:self.end, np.newaxis], v_4_beta_2[self.start:self.end, np.newaxis], v_5_beta_2[self.start:self.end, np.newaxis], v_6_beta_2[self.start:self.end, np.newaxis], v_7_beta_2[self.start:self.end, np.newaxis]))
        elif self.power_nb == 8:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis], v_5[self.start:self.end, np.newaxis], v_6[self.start:self.end, np.newaxis], v_7[self.start:self.end, np.newaxis], v_8[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis], v_5_beta[self.start:self.end, np.newaxis], v_6_beta[self.start:self.end, np.newaxis], v_7_beta[self.start:self.end, np.newaxis], v_8_beta[self.start:self.end, np.newaxis]
                                , v_0_beta_2[self.start:self.end, np.newaxis], v_1_beta_2[self.start:self.end, np.newaxis], v_2_beta_2[self.start:self.end, np.newaxis], v_3_beta_2[self.start:self.end, np.newaxis], v_4_beta_2[self.start:self.end, np.newaxis], v_5_beta_2[self.start:self.end, np.newaxis], v_6_beta_2[self.start:self.end, np.newaxis], v_7_beta_2[self.start:self.end, np.newaxis], v_8_beta_2[self.start:self.end, np.newaxis]))
        return x_beta

    def build_features(self):
        res = []
        for beta in self.beta_list:
            o = self.build_feature_of_beta(beta)
            if len(res) == 0:
                res = o
            else:
                res = np.vstack((res, o))
        return res


class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net,self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False).to(dtype=torch.float64)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs



def train_model(net, x_tensor, y_tensor, epoch):
    initexpr(net)
    opt = torch.optim.Adam(net.parameters())
    lossfunc = torch.nn.MSELoss()
    for t in range(epoch):
        print(t)
        opt.zero_grad()
        prediction = net(x_tensor)
        loss = lossfunc(prediction, y_tensor)
        print(loss)
        loss.backward()
        opt.step()
        if t % 10000 == 0:
            print_model_parameters(net)


def initexpr(model):
    for p in model.parameters():
        p.data = torch.randn(*p.shape, dtype=p.dtype, device=p.device) * 0.1
    return None

def print_model_parameters(linpdelearner):
    for parameters in linpdelearner.parameters():
        print(parameters)

if __name__ == "__main__":
    # 构建数据
    power_nb = 2
    beta_list = [10, 130, 200, 300]
    data = Data(power_nb=power_nb, beta_list=beta_list)
    y = data.build_targets()
    y_tensor = torch.from_numpy(y).to(dtype=torch.float64)
    x = data.build_features()
    x_tensor = torch.from_numpy(x).to(dtype=torch.float64)
    # 引入模型
    # net = Net(27, 1)  # power_nb = 8
    # net = Net(24, 1)  # power_nb = 7
    # net = Net(21, 1) # power_nb = 6
    # net = Net(18, 1)  # power_nb = 5
    # net = Net(15, 1)  # power_nb = 4
    # net = Net(12, 1)  # power_nb = 3
    net = Net(9, 1)  # power_nb = 2
    # net = Net(6, 1)  # power_nb = 1
    # print(net)
    epoch = 200001
    train_model(net, x_tensor, y_tensor, epoch)