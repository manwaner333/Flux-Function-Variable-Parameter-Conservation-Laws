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
    #  grav = -1
    def pd_f_grav_ne_1(self, u):
        molecular = (-4.684781463500322)*u**3+(3.6620276992332177)*u+(-3.259569459538951)*u**2+(3.248647725035878)*u**4+(-1.2969504170619612)*1+(-0.4234111969973986)*u**5+(0.2554915702700467)*u**7+(-0.0807552286699952)*u**8+(-0.03791193425646207)*u**6
        denominator = (-2.9913891670953032)*u**4+(2.712878986769276)*u**3+(2.668585037433123)*u**2+(1.8822193933778693)*u**5+(-1.7852432451296496)*u+(-1.1357545379457716)*u**7+(0.6336770447395432)*1+(0.35898686335463953)*u**8+(0.16853257165614852)*u**6
        return molecular/denominator

    #  grav = 1
    def pd_f_grav_1(self, u):
        molecular = (-13.04888255043475)*u**3+(7.7309690674430644)*u**4+(-6.108933527832838)*u**6+(4.7491329158397715)*u**7+(4.3661971058802)*u+(-2.352255186236733)*1+(-0.9589070576907969)*u**8+(0.7632631220348589)*u**2+(-0.21823744767924805)*u**5
        denominator = (9.325968274119179)*u**6+(8.178466682950976)*u**3+(-7.250081000375348)*u**7+(-7.060440857974147)*u**4+(-2.5812755697172736)*u+(1.931586333217469)*u**2+(1.4638785570524617)*u**8+(1.2347726866413653)*1+(0.3331638008514117)*u**5
        return molecular/denominator

    #  grav = 3
    def pd_f_grav_3(self, u):
        molecular = (7.161660809699515)*u**2+(6.5189899278590335)*u**3+(2.5723895171954183)*u**4+(-1.989732693816193)*u+(0.7345932039010172)*u**5+(0.12256603794465337)*u**6+(-0.067963327370266)*1+(0.00912789687928783)*u**7+(0.00023865702759687135)*u**8
        denominator = (7.709070181263543)*u**3+(5.993486017989935)*u**2+(-5.364296610033849)*u+(3.370906699179923)*u**4+(2.471446089728169)*1+(0.9769730129186811)*u**5+(0.163006832538609)*u**6+(0.012139656163999134)*u**7+(0.0003174021677131518)*u**8
        return molecular/denominator

    # grav = 5
    def pd_f_grav_5(self, u):
        molecular = (8.94967735009678)*u**6+(6.320658319460378)*u**5+(5.292084137960746)*u**7+(2.122328221686811)*u**4+(1.8373929791902128)*u**3+(1.4765712801372972)*u**2+(1.1339768865044928)*u**8+(-0.529989445465226)*1+(-0.3334131388278525)*u
        denominator = (13.77910191079441)*u**6+(9.731411728067858)*u**5+(8.147798384784359)*u**7+(4.167543534244692)*u**3+(3.8412680650740905)*u**4+(-2.7925128250916873)*u+(1.745893452065262)*u**8+(1.4449123811456925)*1+(1.153944584290914)*u**2
        return molecular/denominator

    # grav = 7
    def pd_f_grav_7(self, u):
        molecular = (2.3614964263831437)*u**2+(0.8249119432291696)*u**4+(-0.6629768992834078)*u+(0.33953711136427367)*u**5+(-0.17601311532996683)*1+(-0.16202203397843842)*u**3+(0.04857896744662423)*u**6+(0.003052838196318649)*u**7+(7.226596102077817e-05)*u**8
        denominator = (4.0314836533646465)*u**2+(-3.2001161802335716)*u+(0.9126142102153997)*1+(0.8954866885712938)*u**4+(0.3621482128953715)*u**5+(0.15755364926804083)*u**3+(0.051814030502906956)*u**6+(0.0032561386076863995)*u**7+(7.707843343452301e-05)*u**8
        return molecular/denominator

    # grav = 8
    def pd_f_grav_8(self, u):
        molecular = (17.31317850629746)*u**3+(7.588560883579744)*u**4+(-6.995373340211187)*u+(6.529892105005172)*u**2+(0.789572399386225)*1+(-0.7416135925008617)*u**6+(-0.47106838574152426)*u**7+(0.21847798799996868)*u**5+(-0.08745730561147792)*u**8
        denominator = (14.83157089472402)*u**3+(-8.332976047504077)*u+(5.49154057973625)*u**4+(4.201828760379443)*u**2+(3.108937200483742)*1+(0.009999374075008984)*u**6+(0.006351540817982859)*u**7+(-0.0029457970447925312)*u**5+(0.0011792102022463052)*u**8
        return molecular/denominator


    def buil_traget_of_grav(self, grav):
        if grav == -1:
            d = np.array([self.pd_f_grav_ne_1(ele) - self.pd_f_grav_ne_1(0) for ele in self.x])
        elif grav == 1:
            d = np.array([self.pd_f_grav_1(ele) - self.pd_f_grav_1(0) for ele in self.x])
        elif grav == 3:
            d = np.array([self.pd_f_grav_3(ele) - self.pd_f_grav_3(0) for ele in self.x])
        elif grav == 5:
            d = np.array([self.pd_f_grav_5(ele) - self.pd_f_grav_5(0) for ele in self.x])
        elif grav == 7:
            d = np.array([self.pd_f_grav_7(ele) - self.pd_f_grav_7(0) for ele in self.x])
        elif grav == 8:
            d = np.array([self.pd_f_grav_8(ele) - self.pd_f_grav_8(0) for ele in self.x])
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
    def __init__(self, input_dim, output_dim):
        super(Net,self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False).to(dtype=torch.float64)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def train_model(net, x_tensor, y_tensor, epoch):
    initexpr(net)
    opt = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.MSELoss()
    for t in range(epoch):
        print(t)
        opt.zero_grad()
        prediction = net(x_tensor)
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
    power_nb = 8
    grav_list = [-1, 1, 3, 5, 7]
    data = Data(power_nb=power_nb, grav_list=grav_list)
    y = data.build_targets()
    y_tensor = torch.from_numpy(y).to(dtype=torch.float64)
    x = data.build_features()
    x_tensor = torch.from_numpy(x).to(dtype=torch.float64)
    # # 引入模型
    net = Net(18, 1)
    print(net)
    epoch = 300001
    train_model(net, x_tensor, y_tensor, epoch)
