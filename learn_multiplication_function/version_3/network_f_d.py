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

class Data():
    def __init__(self, power_nb, grav_list):
        self.x = []
        N = 400
        dx = 1.0/N
        for i in range(N + 1):
            self.x.append(i*dx)
        self.start = 0
        self.end = 401
        self.power_nb = power_nb
        self.grav_list = grav_list

    # 给出第一步骤求解出来的表达式
    # #  grav = -1
    # def pd_f_grav_ne_1(self, u):
    #     molecular = (-4.684781463500322)*u**3+(3.6620276992332177)*u+(-3.259569459538951)*u**2+(3.248647725035878)*u**4+(-1.2969504170619612)*1+(-0.4234111969973986)*u**5+(0.2554915702700467)*u**7+(-0.0807552286699952)*u**8+(-0.03791193425646207)*u**6
    #     denominator = (-2.9913891670953032)*u**4+(2.712878986769276)*u**3+(2.668585037433123)*u**2+(1.8822193933778693)*u**5+(-1.7852432451296496)*u+(-1.1357545379457716)*u**7+(0.6336770447395432)*1+(0.35898686335463953)*u**8+(0.16853257165614852)*u**6
    #     return molecular/denominator

    #  grav = 10
    def pd_f_grav_10(self, u):
        molecular = (1.7006642479391998)*u+(-1.6452598353864365)*1+(-0.9640845280843208)*u**3+(0.32141408878787053)*u**4+(-0.019774328349539017)*u**2+(-0.0035837572880650964)*u**6+(-0.0029507141087210752)*u**5+(0.0029023054632591077)*u**7+(-0.00048249102443053484)*u**8
        denominator = (1.0205228295019033)*1+(-0.06627542485672068)*u+(0.043123192645832815)*u**2+(0.002539329424290226)*u**3+(0.0017753486411678139)*u**6+(0.0014617469494484089)*u**5+(-0.0014377659105460562)*u**7+(-0.001274020362540719)*u**4+(0.00023902003281614455)*u**8
        return molecular/denominator


    #  grav = 130
    def pd_f_grav_130(self, u):
        molecular = (-1.2968745891186226)*u**2+(1.2216010064291682)*u+(-0.9677141953929043)*u**3+(0.6587989187866976)*u**4+(-0.5409501142739394)*1+(0.16876825243351062)*u**5+(-0.05534748569220814)*u**6+(-0.045059687236713324)*u**7+(0.01711133819504112)*u**8
        denominator = (0.985669509243963)*u**2+(0.5975083501548867)*u**3+(-0.4796509733311272)*u**4+(-0.37909216023769804)*u+(0.3038064060440057)*1+(0.05726020972853409)*u**5+(-0.018778464509676938)*u**6+(-0.015287988731724553)*u**7+(0.005805587245562908)*u**8
        return molecular/denominator

    # grav = 200
    def pd_f_grav_200(self, u):
        molecular = (-2.260851548587051)*u**2+(1.7718349839154381)*u+(-0.6949712732370981)*1+(-0.3600454113745229)*u**3+(-0.0181190248537432)*u**4+(-0.014385316652640883)*u**5+(-0.00046453569129470634)*u**6+(0.00011901151409314868)*u**7+(-3.004024713501341e-06)*u**8
        denominator = (0.7180772019500903)*u**2+(-0.40927278015013746)*u+(0.2220316882312772)*1+(0.11192666278786897)*u**4+(0.04638811040704861)*u**3+(0.045289444878963955)*u**5+(0.0014625026402418971)*u**6+(-0.0003746852111520395)*u**7+(9.457602843395738e-06)*u**8
        return molecular/denominator


    # grav = 300
    def pd_f_grav_300(self, u):
        molecular = (-3.184890516872588)*u**2+(2.599943994084381)*u+(-0.9848341667546185)*1+(-0.28636677088182)*u**4+(-0.15679871345469737)*u**3+(-0.09813917149173579)*u**5+(-0.007063615029956347)*u**6+(2.001463803709729e-05)*u**7
        denominator = (0.6433489418390006)*u**2+(-0.42933982164898316)*u+(0.20826799467997512)*1+(0.1709331010002923)*u**4+(-0.0886916474469701)*u**3+(0.058492775069557204)*u**5+(0.004210046190984737)*u**6+(-1.1929097250440108e-05)*u**7
        return molecular/denominator


    def buil_traget_of_grav(self, grav):
        if grav == 10:
            d = np.array([self.pd_f_grav_10(ele) - self.pd_f_grav_10(0) for ele in self.x])
        elif grav == 130:
            d = np.array([self.pd_f_grav_130(ele) - self.pd_f_grav_130(0) for ele in self.x])
        elif grav == 200:
            d = np.array([self.pd_f_grav_200(ele) - self.pd_f_grav_200(0) for ele in self.x])
        elif grav == 300:
            d = np.array([self.pd_f_grav_300(ele) - self.pd_f_grav_300(0) for ele in self.x])
        return d

    def build_targets(self):
        res = []
        for grav in self.grav_list:
            d = self.buil_traget_of_grav(grav)
            if len(res) == 0:
                res = d[self.start:self.end, np.newaxis]
            else:
                res = np.vstack((res, d[self.start:self.end, np.newaxis]))
        return res


    # 构建特征x
    # 假设函数的表示式：
    # 分母部分： a0 + a1*u + a2*u**2 + a3*u**3 + a4*u**4 + a5*u**5 + a6*u**6 + grav *(b0 + b1*u + b2*u**2 + b3* u**3 + b4*u**4 + b5*u**5 + b6*u**6)
    # 分子部分： c0 + c1*u + c2*u**2 + c3*u**3 + c4*u**4 + c5*u**5 + c6*u**6 + grav *(d0 + d1*u + d2*u**2 + d3* u**3 + d4*u**4 + d5*u**5 + d6*u**6)
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
    def v_beta_0(self, u, grav):
        v = grav
        return v
    def v_beta_1(self, u, grav):
        v = grav * u
        return v
    def v_beta_2(self, u, grav):
        v = grav * u**2
        return v
    def v_beta_3(self, u, grav):
        v = grav * u**3
        return v
    def v_beta_4(self, u, grav):
        v = grav * u**4
        return v
    def v_beta_5(self, u, grav):
        v = grav * u**5
        return v
    def v_beta_6(self, u, grav):
        v = grav * u**6
        return v
    def v_beta_7(self, u, grav):
        v = grav * u**7
        return v
    def v_beta_8(self, u, grav):
        v = grav * u**8
        return v

    def build_feature_of_grav(self, grav):
        v_0 = np.array([self.v_0(ele) for ele in self.x])
        v_1 = np.array([self.v_1(ele) for ele in self.x])
        v_2 = np.array([self.v_2(ele) for ele in self.x])
        v_3 = np.array([self.v_3(ele) for ele in self.x])
        v_4 = np.array([self.v_4(ele) for ele in self.x])
        v_5 = np.array([self.v_5(ele) for ele in self.x])
        v_6 = np.array([self.v_6(ele) for ele in self.x])
        v_7 = np.array([self.v_7(ele) for ele in self.x])
        v_8 = np.array([self.v_8(ele) for ele in self.x])

        v_0_beta = np.array([self.v_beta_0(ele, grav) for ele in self.x])
        v_1_beta = np.array([self.v_beta_1(ele, grav) for ele in self.x])
        v_2_beta = np.array([self.v_beta_2(ele, grav) for ele in self.x])
        v_3_beta = np.array([self.v_beta_3(ele, grav) for ele in self.x])
        v_4_beta = np.array([self.v_beta_4(ele, grav) for ele in self.x])
        v_5_beta = np.array([self.v_beta_5(ele, grav) for ele in self.x])
        v_6_beta = np.array([self.v_beta_6(ele, grav) for ele in self.x])
        v_7_beta = np.array([self.v_beta_7(ele, grav) for ele in self.x])
        v_8_beta = np.array([self.v_beta_8(ele, grav) for ele in self.x])

        if self.power_nb == 1:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis]))
        elif self.power_nb == 2:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis]))
        elif self.power_nb == 3:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis]))
        elif self.power_nb == 4:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis]
                                    , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis]))
        elif self.power_nb == 5:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis], v_5[self.start:self.end, np.newaxis]
                            , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis], v_5_beta[self.start:self.end, np.newaxis]))
        elif self.power_nb == 6:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis], v_5[self.start:self.end, np.newaxis], v_6[self.start:self.end, np.newaxis]
                                  , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis], v_5_beta[self.start:self.end, np.newaxis], v_6_beta[self.start:self.end, np.newaxis]))
        elif self.power_nb == 7:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis], v_5[self.start:self.end, np.newaxis], v_6[self.start:self.end, np.newaxis], v_7[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis], v_5_beta[self.start:self.end, np.newaxis], v_6_beta[self.start:self.end, np.newaxis], v_7_beta[self.start:self.end, np.newaxis]))
        elif self.power_nb == 8:
            x_beta = np.hstack((v_0[self.start:self.end, np.newaxis], v_1[self.start:self.end, np.newaxis], v_2[self.start:self.end, np.newaxis], v_3[self.start:self.end, np.newaxis], v_4[self.start:self.end, np.newaxis], v_5[self.start:self.end, np.newaxis], v_6[self.start:self.end, np.newaxis], v_7[self.start:self.end, np.newaxis], v_8[self.start:self.end, np.newaxis]
                                , v_0_beta[self.start:self.end, np.newaxis], v_1_beta[self.start:self.end, np.newaxis], v_2_beta[self.start:self.end, np.newaxis], v_3_beta[self.start:self.end, np.newaxis], v_4_beta[self.start:self.end, np.newaxis], v_5_beta[self.start:self.end, np.newaxis], v_6_beta[self.start:self.end, np.newaxis], v_7_beta[self.start:self.end, np.newaxis], v_8_beta[self.start:self.end, np.newaxis]))
        return x_beta

    def build_features(self):
        res = []
        for grav in self.grav_list:
            o = self.build_feature_of_grav(grav)
            if len(res) == 0:
                res = o
            else:
                res = np.vstack((res, o))
        return res

class Net(torch.nn.Module):
    def __init__(self, input_molecular_nb, output_molecular_nb, input_denominator_nb, output_denominator_nb):
        super(Net,self).__init__()
        # 分子
        self.molecular = torch.nn.Linear(input_molecular_nb, output_molecular_nb, bias=False).to(dtype=torch.float64)
        # 分母
        self.denominator = torch.nn.Linear(input_denominator_nb, output_denominator_nb, bias=False).to(dtype=torch.float64)

    def forward(self, molecular_features, denominator_features):
        molecular = self.molecular(molecular_features)
        denominator = self.denominator(denominator_features)
        y = molecular/denominator
        return y


def train_model(net, molecular_x_tensor, denominator_x_tensor, y_tensor, epoch):
    initexpr(net)
    opt = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.MSELoss()
    for t in range(epoch):
        print(t)
        opt.zero_grad()
        prediction = net(molecular_x_tensor, denominator_x_tensor)
        loss = loss_func(prediction, y_tensor)
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
    power_nb = 4   # 5
    grav_list = [10, 130, 200, 300]
    data = Data(power_nb=power_nb, grav_list=grav_list)
    y = data.build_targets()
    y_tensor = torch.from_numpy(y).to(dtype=torch.float64)
    molecular_x = data.build_features()
    denominator_x = data.build_features()
    molecular_x_tensor = torch.from_numpy(molecular_x).to(dtype=torch.float64)
    denominator_x_tensor = torch.from_numpy(denominator_x).to(dtype=torch.float64)
    # # 引入模型
    net = Net(10, 1, 10, 1)
    print(net)
    epoch = 200001
    train_model(net, molecular_x_tensor, denominator_x_tensor, y_tensor, epoch)
