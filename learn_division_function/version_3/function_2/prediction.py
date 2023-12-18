from numpy import *
import numpy as np
import torch
from torch.autograd import Variable
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# from torchviz import make_dot
# import expr
from torch.autograd import grad
__all__ = ['VariantCoeLinear1d']


class VariantCoeLinear1d(torch.nn.Module):
    def __init__(self, T, N, X, batch_size, u0, dt, time_steps, dx, max_f_prime, u_fixed, device, beta, power_nb, is_real, is_train=True):
        super(VariantCoeLinear1d, self).__init__()
        coe_num = 1  # the number of coefficient
        self.coe_num = coe_num
        self.T = T
        self.N = N  # The number of grid cell
        self.X = X
        self.batch_size = batch_size
        self.allchannels = ['u']
        self.channel_num = 1
        self.hidden_layers = 3
        self.register_buffer('u0', u0)
        self.register_buffer('u_fixed', u_fixed)
        self.register_buffer('dt', torch.DoubleTensor(1).fill_(dt))
        self.register_buffer('dx', torch.DoubleTensor(1).fill_(dx))
        self.register_buffer('max_f_prime', torch.DoubleTensor(1).fill_(max_f_prime))
        self.register_buffer('time_steps', torch.DoubleTensor(1).fill_(time_steps))
        self.device = device
        self.beta = beta
        self.power_nb = power_nb
        self.is_real = is_real
        self.is_train = is_train

    @property
    def coes(self):
        for i in range(self.coe_num):
            yield self.__getattr__('coe'+str(i))
    @property
    def xy(self):
        return Variable(next(self.coes).inputs)
    @xy.setter
    def xy(self, v):
        for fitter in self.coes:
            fitter.inputs = v

    def coe_params(self):
        parameters = []
        for poly in self.polys:
            parameters += list(poly.parameters())
        return parameters

    def f_predict(self, u):
        u = u.unsqueeze(1)
        Uadd = list(poly(u.permute(0, 2, 1)) for poly in self.polys)
        uadd = torch.cat(Uadd, dim=1)
        return uadd

    def f_real(self, u):
        if is_real:
            f = (torch.pow(u, 2) * (1 - self.beta * torch.pow(1-u, 4))) / (torch.pow(u, 2) + 0.5 * torch.pow(1-u, 4))
            return f
        else:
            if self.power_nb == 8:
                molecular_coe = [ 0.0297, -0.2006, -0.2684,  0.0892,  0.2973, -0.1894, -0.7679, -0.8349,
                                  0.7015,  0.0093,  0.1088, -0.5055, -0.1162,  0.6518,  0.8877,  0.6654,
                                  -0.1231, -0.7964]
                denominator_coe = [-0.1775,  0.1747, -0.2943, -0.0108, -0.0834, -0.2282, -0.1190, -0.1271,
                                   -0.3280, -0.0847, -0.0100,  0.1639,  0.1523,  0.1541,  0.0890,  0.1646,
                                   0.1449,  0.0945]
                mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) + molecular_coe[6] * torch.pow(u, 6) + molecular_coe[7] * torch.pow(u, 7) + molecular_coe[8] * torch.pow(u, 8) \
                       + self.beta * (molecular_coe[9] + molecular_coe[10] * u + molecular_coe[11] * torch.pow(u, 2) + molecular_coe[12] * torch.pow(u, 3) + molecular_coe[13] * torch.pow(u, 4) + molecular_coe[14] * torch.pow(u, 5) + molecular_coe[15] * torch.pow(u, 6) + molecular_coe[16] * torch.pow(u, 7) + molecular_coe[17] * torch.pow(u, 8))
                deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) + denominator_coe[6] * torch.pow(u, 6) + denominator_coe[7] * torch.pow(u, 7) + denominator_coe[8] * torch.pow(u, 8) \
                       + self.beta * (denominator_coe[9] + denominator_coe[10] * u + denominator_coe[11] * torch.pow(u, 2) + denominator_coe[12] * torch.pow(u, 3) + denominator_coe[13] * torch.pow(u, 4) + denominator_coe[14] * torch.pow(u, 5) + denominator_coe[15] * torch.pow(u, 6) + denominator_coe[16] * torch.pow(u, 7) + denominator_coe[17] * torch.pow(u, 8))
                f = mole/deno
                return f
            elif self.power_nb == 7:
                molecular_coe = [2.3122e-03, -2.3222e-01,  5.6972e-01,  5.9203e-01,  1.9172e-01,
                                  -5.7805e-03,  2.4819e-01,  4.0770e-01,  1.4875e-03, -2.2347e-01,
                                  6.3686e-01,  4.2491e-01,  2.1280e-01,  2.4168e-01,  4.2672e-01,
                                  1.8061e+00]
                denominator_coe = [0.1497, -0.0508, -0.1466,  0.2429,  0.7619,  0.7545,  0.3501, -0.3188,
                                    0.1725, -0.1388, -0.0269,  0.3828,  0.3654,  0.8305,  0.9993,  0.9177]
                mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) + molecular_coe[6] * torch.pow(u, 6) + molecular_coe[7] * torch.pow(u, 7) \
                       + self.beta * (molecular_coe[8] + molecular_coe[9] * u + molecular_coe[10] * torch.pow(u, 2) + molecular_coe[11] * torch.pow(u, 3) + molecular_coe[12] * torch.pow(u, 4) + molecular_coe[13] * torch.pow(u, 5) + molecular_coe[14] * torch.pow(u, 6) + molecular_coe[15] * torch.pow(u, 7))
                deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) + denominator_coe[6] * torch.pow(u, 6) + denominator_coe[7] * torch.pow(u, 7) \
                       + self.beta * (denominator_coe[8] + denominator_coe[9] * u + denominator_coe[10] * torch.pow(u, 2) + denominator_coe[11] * torch.pow(u, 3) + denominator_coe[12] * torch.pow(u, 4) + denominator_coe[13] * torch.pow(u, 5) + denominator_coe[14] * torch.pow(u, 6) + denominator_coe[15] * torch.pow(u, 7))
                f = mole/deno
                return f
            elif self.power_nb == 6:
                molecular_coe = [-7.0699e-04,  1.0301e-01, -1.2928e+00, -1.0959e+00, -2.8249e-01,
                                 -3.5936e+00, -7.7790e+00,  1.4742e-04,  3.7239e-02,  2.4613e-01,
                                 4.2619e-01, -2.4086e+00,  3.4410e+00, -3.0852e+00]
                denominator_coe = [-2.6457e-01,  1.3283e-01, -1.2820e-01, -1.7147e+00, -1.6430e+00,
                                   -2.1634e+00, -8.1376e+00,  2.6369e-03, -9.7678e-02,  4.7533e-01,
                                   -6.4356e-01,  7.3440e-02,  1.4473e+00, -2.6043e+00]
                mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) + molecular_coe[6] * torch.pow(u, 6) \
                       + self.beta * (molecular_coe[7] + molecular_coe[8] * u + molecular_coe[9] * torch.pow(u, 2) + molecular_coe[10] * torch.pow(u, 3) + molecular_coe[11] * torch.pow(u, 4) + molecular_coe[12] * torch.pow(u, 5) + molecular_coe[13] * torch.pow(u, 6))
                deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) + denominator_coe[6] * torch.pow(u, 6) \
                       + self.beta * (denominator_coe[7] + denominator_coe[8] * u + denominator_coe[9] * torch.pow(u, 2) + denominator_coe[10] * torch.pow(u, 3) + denominator_coe[11] * torch.pow(u, 4) + denominator_coe[12] * torch.pow(u, 5) + denominator_coe[13] * torch.pow(u, 6))
                f = mole/deno
                return f
            elif self.power_nb == 5:
                molecular_coe = [2.9257e-04, -6.0451e-02,  8.5192e-01, -1.1258e-01,  3.8009e-01,
                                  1.9196e+00, -1.1714e-04, -2.1085e-02, -1.8447e-01,  2.6088e-01,
                                  7.7523e-02,  9.9530e-02]
                denominator_coe = [1.7557e-01, -3.1120e-01,  4.8654e-01,  5.0932e-01,  4.0039e-01,
                                    1.7099e+00, -1.6422e-03,  4.8316e-02, -1.8631e-01,  6.4508e-02,
                                    1.8323e-01,  1.2594e-01]
                mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) \
                       + self.beta * (molecular_coe[6] + molecular_coe[7] * u + molecular_coe[8] * torch.pow(u, 2) + molecular_coe[9] * torch.pow(u, 3) + molecular_coe[10] * torch.pow(u, 4) + molecular_coe[11] * torch.pow(u, 5))
                deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) \
                       + self.beta * (denominator_coe[6] + denominator_coe[7] * u + denominator_coe[8] * torch.pow(u, 2) + denominator_coe[9] * torch.pow(u, 3) + denominator_coe[10] * torch.pow(u, 4) + denominator_coe[11] * torch.pow(u, 5))
                f = mole/deno
                return f
            elif self.power_nb == 4:
                molecular_coe = [1.4001e-03, -2.2150e-01,  3.1953e+00, -2.2674e+00,  4.6245e+00,
                                 -6.2421e-05, -1.0880e-01, -3.2954e-01,  2.8313e-01,  9.3187e-01]
                denominator_coe = [6.1315e-01, -1.1811e+00,  1.4468e+00,  1.9004e+00,  2.5291e+00,
                                    1.2177e-02,  2.6833e-04, -9.8689e-02, -6.5726e-01,  1.5263e+00]
                mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) \
                       + self.beta * (molecular_coe[5] + molecular_coe[6] * u + molecular_coe[7] * torch.pow(u, 2) + molecular_coe[8] * torch.pow(u, 3) + molecular_coe[9] * torch.pow(u, 4))
                deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) \
                       + self.beta * (denominator_coe[5] + denominator_coe[6] * u + denominator_coe[7] * torch.pow(u, 2) + denominator_coe[8] * torch.pow(u, 3) + denominator_coe[9] * torch.pow(u, 4))
                f = mole/deno
                return f
            elif self.power_nb == 3:
                molecular_coe = [0.0074, -0.3308,  1.1144, -0.1477,  0.0072, -0.3240,  1.0923, -0.0537]
                denominator_coe = [0.1489, -0.1299, -0.0212,  0.6736,  0.1567, -0.1505, -0.0136,  0.7577]
                mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) \
                       + self.beta * (molecular_coe[4] + molecular_coe[5] * u + molecular_coe[6] * torch.pow(u, 2) + molecular_coe[7] * torch.pow(u, 3))
                deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) \
                       + self.beta * (denominator_coe[4] + denominator_coe[5] * u + denominator_coe[6] * torch.pow(u, 2) + denominator_coe[7] * torch.pow(u, 3) )
                f = mole/deno
                return f

    def f_half(self, u):
        if self.is_train:
            f = self.f_predict(u)
        else:
            f = self.f_real(u)
        f_half = torch.empty((self.batch_size, self.N - 1), requires_grad=False).to(self.device)
        for index in range(self.N - 1):
            b = u[:, index:index+2].clone().detach()
            b.requires_grad = True
            dfdu = self.df_du(b)
            b.requires_grad = False
            # f_half[:, index] = 0.5 * (f[:, index] + f[:, index + 1]) - 0.5 * torch.max(dfdu) * (u[:, index + 1] - u[:, index])
            f_half[:, index] = 0.5 * (f[:, index] + f[:, index + 1]) - 0.5 * torch.max(dfdu, 1).values * (u[:, index + 1] - u[:, index])
        return f_half

    # 没有系数
    def a(self):
        x = []
        for index in range(self.N):
            if index == 0:
                x.append(self.dx * 0.5)
            else:
                x.append(self.dx * 0.5 + index * self.dx)
        # true parameters: 0.5 * x
        res = torch.empty((1, self.N), requires_grad=False).to(self.device)
        for index in range(self.N):
            res[:, index] = 1.0
        return res.repeat(self.batch_size, 1).double()

    def df_du(self, u):
        if self.is_train:
            f = self.f_predict(u)
        else:
            f = self.f_real(u)
        # 计算目前f(u)下面的导数
        dfdu = grad(f, u, grad_outputs=torch.ones_like(f), create_graph=False)[0]
        dfdu = torch.abs(dfdu)
        return dfdu

    def update(self):
        # 计算目前状况下f(u)导数的最大值
        self.u_fixed.requires_grad = True
        dfdu = self.df_du(self.u_fixed)
        max_f_prime = torch.max(dfdu).item()
        self.u_fixed.requires_grad = False
        if max_f_prime > 0 and max_f_prime < 100:
            dt_a = 0.75 * self.dx.item()/(max_f_prime + 0.0001)
            n_time = self.T/dt_a
            n_time = int(round(n_time+1, 0))
            dt = self.T/n_time
            self.max_f_prime = torch.DoubleTensor(1).fill_(max_f_prime).to(self.device)
            self.dt = torch.DoubleTensor(1).fill_(dt).to(self.device)
            self.time_steps = torch.IntTensor(1).fill_(n_time).to(self.device)
            print("is_real")
            print(self.is_real)
            print("\033[34mbeta %.6f, power_nb %.6f, max_f_prime %.6f, dt %.6f, time_steps %.6f,\033[0m" % (self.beta, self.power_nb, self.max_f_prime, self.dt, self.time_steps))


    def forward(self, init, stepnum):
        u_old = init
        dt = self.dt
        dx = self.dx
        coefficient = self.a()
        trajectories = torch.empty((stepnum, self.batch_size, self.N), requires_grad=False, device=self.device)
        trajectories[0, :, :] = u_old
        for i in range(1, stepnum):
            f_half = self.f_half(u_old)
            u = torch.empty((self.batch_size, self.N), requires_grad=False).to(self.device)
            for j in range(1, self.N - 1):
                u[:, j] = u_old[:, j] - coefficient[:, j] * (dt/dx) * (f_half[:, j] - f_half[:, j-1])
            u[:, 0] = u[:, 1]
            u[:, self.N - 1] = u[:, self.N - 2]
            u_old = u
            # print(u_old.min(), u_old.max())
            trajectories[i, :] = u_old
        return trajectories


def generate_real_data(save_file, beta, power_nb, is_real):
    device = 'cpu'
    T = 2   # 40
    X = 10
    dt = 0.08
    dx = 0.025  # 0.0125  # 0.025   # 0.05
    N = 400  # 800  # 400   # 200
    time_steps = 200
    max_f_prime = -0.03
    theta = 0.001
    # 之前用到的
    # u_0
    batch_size = 4
    u_0_np = np.zeros((batch_size, N), dtype=float)
    u_0_np[:1, 180:240] = 1.0
    u_0_np[1:2, 140:200] = 0.9
    u_0_np[2:3, 200:260] = 0.8
    u_0_np[3:4, 120:240] = 0.7
    u_0 = torch.from_numpy(u_0_np)
    u_0 = u_0.to(device)
    # # 测试局部numerical scheme 的时候用到的
    # batch_size = 6
    # u_0_np = np.zeros((batch_size, N), dtype=float)
    # u_0_np[:1, 180:240] = 1.0
    # u_0_np[1:2, 200:260] = 0.8
    # u_0_np[1:2, 0:200] = 0.2
    # u_0_np[1:2, 260:400] = 0.2
    # u_0_np[2:3, 160:280] = 0.85
    # u_0_np[3:4, 0:120] = 0.95
    # u_0_np[4:5, 120:240] = 0.7
    # u_0_np[5:6, 140:200] = 0.9
    # u_0 = torch.from_numpy(u_0_np)
    # u_0 = u_0.to(device)
    # batch_size = 1
    # u_0_np = np.zeros((batch_size, N), dtype=float)
    # u_0_np[:1, 180:240] = 1.0
    # u_0 = torch.from_numpy(u_0_np)
    # u_0 = u_0.to(device)
    du = 1.0/500
    u_fixed_0 = 0.0
    u_fixed_np = np.zeros((1, 501), dtype=float)
    u_fixed_np[:1, 0] = u_fixed_0
    for i in range(1, 501):
        u_fixed_np[:1, i] = u_fixed_0 + i * du
    u_fixed = torch.from_numpy(u_fixed_np)
    u_fixed = u_fixed.to(device)
    # model
    linpdelearner = VariantCoeLinear1d(T=T, N=N, X=X, batch_size=batch_size, u0=u_0, dt=dt, time_steps=time_steps
                                       , dx=dx, max_f_prime=max_f_prime, u_fixed=u_fixed, device=device, beta=beta, power_nb=power_nb, is_real=is_real, is_train=False)
    # 预测值
    linpdelearner.update()
    U = linpdelearner(linpdelearner.u0, linpdelearner.time_steps)
    print("U")
    print(U.shape)
    np.save(save_file, U.detach().to('cpu'))


if __name__ == "__main__":
    beta_list = [-3]    # [-3, -1, 9]
    for power_nb in [8]:  # [3, 4, 5, 6, 7]
        for beta in beta_list:
            for is_real in [True, False]:
                if beta < 0:
                    beta_name = 'ne_' + str(np.abs(beta))
                else:
                    beta_name = str(beta)
                experiment_name = 'N_400_example_4_beta_' + beta_name + '_power_np_' + str(power_nb)
                save_dir = 'data_d/' + experiment_name + '/'
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                if is_real:
                    real_data_file = '/' + experiment_name + '_real_U' + '.npy'
                    generate_real_data(save_dir + real_data_file, beta, power_nb, is_real)
                else:
                    real_data_file = '/' + experiment_name + '_predict_U' + '.npy'
                    generate_real_data(save_dir + real_data_file, beta, power_nb, is_real)




