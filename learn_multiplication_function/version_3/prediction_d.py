from numpy import *
import torch
from torch.autograd import Variable
# from torchviz import make_dot
# import expr
from torch.autograd import grad
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.random.seed(0)
torch.manual_seed(0)
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
            f = 0.5 * u * (3 - torch.pow(u, 2)) + (1/12) * self.beta * torch.pow(u, 2) * ((3/4) - 2*u +(3/2)*torch.pow(u, 2) - (1/4)*torch.pow(u, 4))
            return f
        else:
            # if self.power_nb == 8:
            #     coe = [1.6546e-02,  1.2972e+00,  3.4795e-01, -3.4739e-01, -2.9556e-01, -1.0908e-01, -1.7838e-01,  1.7805e-02,  2.5824e-01, -1.9990e-04,
            #            8.0966e-03, -1.9496e-03,  3.7383e-02, -1.4915e-01,  1.0300e-01, 3.3892e-02, -1.8826e-03, -2.9432e-02]
            #     f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) + coe[3] * torch.pow(u, 3) + coe[4] * torch.pow(u, 4) + coe[5] * torch.pow(u, 5) + coe[6] * torch.pow(u, 6) + coe[7] * torch.pow(u, 7) + coe[8] * torch.pow(u, 8) \
            #         + self.beta * (coe[9] + coe[10] * u + coe[11] * torch.pow(u, 2) + coe[12] * torch.pow(u, 3) + coe[13] * torch.pow(u, 4) + coe[14] * torch.pow(u, 5) + coe[15] * torch.pow(u, 6) + coe[16] * torch.pow(u, 7) + coe[17] * torch.pow(u, 8))
            #     return f
            # elif self.power_nb == 7:
            #     coe = [1.2900e-02,  1.3306e+00,  2.9959e-01, -3.2564e-01, -3.7582e-01, -2.3260e-01,  1.6571e-01,  1.2841e-01, -6.3185e-05,  3.5757e-03,
            #            3.6194e-02, -9.4567e-02,  6.6900e-02, -7.4482e-02,  1.2975e-01, -6.7465e-02]
            #     f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) + coe[3] * torch.pow(u, 3) + coe[4] * torch.pow(u, 4) + coe[5] * torch.pow(u, 5) + coe[6] * torch.pow(u, 6) + coe[7] * torch.pow(u, 7) \
            #         + self.beta * (coe[8] + coe[9] * u + coe[10] * torch.pow(u, 2) + coe[11] * torch.pow(u, 3) + coe[12] * torch.pow(u, 4) + coe[13] * torch.pow(u, 5) + coe[14] * torch.pow(u, 6) + coe[15] * torch.pow(u, 7))
            #     return f
            # elif self.power_nb == 6:
            #     coe = [8.4920e-03,  1.3823e+00,  1.9598e-01, -3.3295e-01, -3.9308e-01, -1.8213e-02,  1.5651e-01, -1.7243e-05,  2.0499e-03,  4.8243e-02,
            #            -1.2439e-01,  5.9704e-02,  5.1649e-02, -3.7291e-02]
            #     f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) + coe[3] * torch.pow(u, 3) + coe[4] * torch.pow(u, 4) + coe[5] * torch.pow(u, 5) + coe[6] * torch.pow(u, 6) \
            #         + self.beta * (coe[7] + coe[8] * u + coe[9] * torch.pow(u, 2) + coe[10] * torch.pow(u, 3) + coe[11] * torch.pow(u, 4) + coe[12] * torch.pow(u, 5) + coe[13] * torch.pow(u, 6))
            #     return f
            if self.power_nb == 5:
                molecular_coe = [-2.9963e-03,  3.2242e+00,  2.4224e+00,  2.8889e+00,  2.0823e+00, -2.5585e-01,  1.3445e-04,  1.0289e-02,  8.0752e-02, -1.9913e-01,
                                 8.6455e-02,  4.6917e-02]
                denominator_coe = [2.1802e+00,  1.6735e+00,  1.8975e+00,  2.2489e+00,  1.8277e+00, 4.6739e-01,  2.0961e-03, -6.1652e-03, -2.7121e-02,  9.1336e-02,
                                   -1.2471e-01,  9.0318e-02]
                mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) \
                       + self.beta * (molecular_coe[6] + molecular_coe[7] * u + molecular_coe[8] * torch.pow(u, 2) + molecular_coe[9] * torch.pow(u, 3) + molecular_coe[10] * torch.pow(u, 4) + molecular_coe[11] * torch.pow(u, 5))
                deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) \
                       + self.beta * (denominator_coe[6] + denominator_coe[7] * u + denominator_coe[8] * torch.pow(u, 2) + denominator_coe[9] * torch.pow(u, 3) + denominator_coe[10] * torch.pow(u, 4) + denominator_coe[11] * torch.pow(u, 5))
                f = mole/deno
                return f
            elif self.power_nb == 4:
                molecular_coe = [3.5963e-02, -7.8481e+00, -3.0674e-01, -1.9213e+01, -6.2573e+00,
                                 -6.6967e-04,  9.6586e-04, -3.6481e-01,  9.2113e-01, -6.5839e-01]
                denominator_coe = [-4.7379e+00, -3.9272e+00, -5.5947e+00, -7.5931e+00, -1.1535e+01,
                                   -4.2372e-03,  3.1649e-02, -1.4368e-01,  4.3875e-01, -4.2753e-01]
                mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) \
                       + self.beta * (molecular_coe[5] + molecular_coe[6] * u + molecular_coe[7] * torch.pow(u, 2) + molecular_coe[8] * torch.pow(u, 3) + molecular_coe[9] * torch.pow(u, 4))
                deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) \
                       + self.beta * (denominator_coe[5] + denominator_coe[6] * u + denominator_coe[7] * torch.pow(u, 2) + denominator_coe[8] * torch.pow(u, 3) + denominator_coe[9] * torch.pow(u, 4))
                f = mole/deno
                return f
            elif self.power_nb == 3:
                molecular_coe = [2.6862e-03,  2.7903e+00,  2.5055e+00,  1.1115e+00, -1.4754e-04,
                                 3.3459e-02, -6.3614e-02,  3.8542e-02]
                denominator_coe = [2.0864,  0.9397,  1.1844,  2.1560,  0.0084, -0.0451,  0.0576, -0.0117]
                mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) \
                       + self.beta * (molecular_coe[4] + molecular_coe[5] * u + molecular_coe[6] * torch.pow(u, 2) + molecular_coe[7] * torch.pow(u, 3))
                deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3)  \
                       + self.beta * (denominator_coe[4] + denominator_coe[5] * u + denominator_coe[6] * torch.pow(u, 2) + denominator_coe[7] * torch.pow(u, 3))
                f = mole/deno

                return f
            # elif self.power_nb == 2:
            #     coe = [-2.5740e-02,  1.7862e+00, -7.3475e-01, -2.3229e-05,  9.3153e-03, -1.0293e-02]
            #     f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) \
            #         + self.beta * (coe[3] + coe[4] * u + coe[5] * torch.pow(u, 2))
            #     return f
            # elif self.power_nb == 1:
            #     coe = [9.6418e-02,  1.0514e+00,  1.6945e-03, -9.6213e-04]
            #     f = coe[0] + coe[1] * u + self.beta * (coe[2] + coe[3] * u)
            #     return f


    def f_half(self, u):
        if self.is_train:
            f = 1.0 * self.f_predict(u)
        else:
            f = 1.0 * self.f_real(u)
        f_half = torch.empty((self.batch_size, self.N - 1), requires_grad=False).to(self.device)
        for index in range(self.N - 1):
            b = u[:, index:index+2].clone().detach()
            b.requires_grad = True
            dfdu = self.df_du(b)
            b.requires_grad = False
            f_half[:, index] = 0.5 * (f[:, index] + f[:, index + 1]) - 0.5 * torch.max(dfdu, 1).values * (u[:, index + 1] - u[:, index])
        return f_half

    def df_du(self, u):
        if self.is_train:
            f = 1.0 * self.f_predict(u)
        else:
            f = 1.0 * self.f_real(u)
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
        trajectories = torch.empty((stepnum, self.batch_size, self.N), requires_grad=False, device=self.device)
        trajectories[0, :, :] = u_old
        for i in range(1, stepnum):
            f_half = self.f_half(u_old)
            u = torch.empty((self.batch_size, self.N), requires_grad=False).to(self.device)
            for j in range(1, self.N - 1):
                u[:, j] = u_old[:, j] - (dt/dx) * (f_half[:, j] - f_half[:, j-1])
            u[:, 0] = u[:, 1]
            u[:, self.N - 1] = u[:, self.N - 2]
            u_old = u
            trajectories[i, :] = u_old
        return trajectories

# generate real data
def generate_real_data(save_file, beta, power_nb, is_real):
    device = 'cpu'
    T = 2.0
    X = 10
    N = 400  # 200
    dx = X/N  # 10/200   # 10/1600  # 2   # 0.05
    dt = 0.023529
    time_steps = 200
    max_f_prime = -0.03
    # 初始状态
    batch_size = 4
    u_0_np = np.zeros((batch_size, N), dtype=float)
    u_0_np[:1, 180:240] = 1.0
    u_0_np[1:2, 140:200] = 0.9
    u_0_np[2:3, 200:260] = 0.8
    u_0_np[3:4, 120:240] = 0.7
    u_0 = torch.from_numpy(u_0_np)
    u_0 = u_0.to(device)
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
    # [1, 2, 3, 4, 5, 6, 7, 8]
    # beta_list = [-150, -100, -50, 350, 400]
    beta_list = [-50, 350]
    for power_nb in [3, 4]:
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





