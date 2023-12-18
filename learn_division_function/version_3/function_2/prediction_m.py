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
                coe =[ 6.3088e-03, -8.9215e-01,  1.0290e+01, -5.1067e+00, -1.2334e+01,
                       -1.6851e+00,  1.1300e+01,  1.1168e+01, -1.1770e+01,  9.6516e-03,
                       -2.6369e-01, -1.9240e+00,  6.8170e+00, -3.1741e+00, -5.0403e+00,
                       9.1369e-01,  4.9319e+00, -2.2626e+00]
                f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) + coe[3] * torch.pow(u, 3) + coe[4] * torch.pow(u, 4) + coe[5] * torch.pow(u, 5) + coe[6] * torch.pow(u, 6) + coe[7] * torch.pow(u, 7) + coe[8] * torch.pow(u, 8) \
                    + self.beta * (coe[9] + coe[10] * u + coe[11] * torch.pow(u, 2) + coe[12] * torch.pow(u, 3) + coe[13] * torch.pow(u, 4) + coe[14] * torch.pow(u, 5) + coe[15] * torch.pow(u, 6) + coe[16] * torch.pow(u, 7) + coe[17] * torch.pow(u, 8))
                return f
            elif self.power_nb == 7:
                coe = [ 8.2681e-04, -9.0358e-01,  1.1305e+01, -8.9121e+00, -1.1344e+01,
                        5.4914e+00,  1.4598e+01, -9.2226e+00,  1.4147e-02, -3.8340e-01,
                        -1.2327e+00,  5.7058e+00, -3.7262e+00, -3.3704e+00,  3.0961e+00,
                        -9.4790e-02]
                f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) + coe[3] * torch.pow(u, 3) + coe[4] * torch.pow(u, 4) + coe[5] * torch.pow(u, 5) + coe[6] * torch.pow(u, 6) + coe[7] * torch.pow(u, 7) \
                    + self.beta * (coe[8] + coe[9] * u + coe[10] * torch.pow(u, 2) + coe[11] * torch.pow(u, 3) + coe[12] * torch.pow(u, 4) + coe[13] * torch.pow(u, 5) + coe[14] * torch.pow(u, 6) + coe[15] * torch.pow(u, 7))
                return f
            elif self.power_nb == 6:
                coe = [ -0.0181,  -0.5221,  10.0918, -10.0130,  -6.1992,   9.4225,  -1.7064,
                        0.0173,  -0.4678,  -0.7494,   4.8684,  -3.7966,  -2.0554,   2.1886]
                f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) + coe[3] * torch.pow(u, 3) + coe[4] * torch.pow(u, 4) + coe[5] * torch.pow(u, 5) + coe[6] * torch.pow(u, 6) \
                    + self.beta * (coe[7] + coe[8] * u + coe[9] * torch.pow(u, 2) + coe[10] * torch.pow(u, 3) + coe[11] * torch.pow(u, 4) + coe[12] * torch.pow(u, 5) + coe[13] * torch.pow(u, 6))
                return f
            elif self.power_nb == 5:
                coe = [-0.0224, -0.3206,  8.5242, -6.1717, -8.7515,  7.8176,  0.0180, -0.4742,
                       -0.9457,  6.6184, -8.6716,  3.4531]
                f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) + coe[3] * torch.pow(u, 3) + coe[4] * torch.pow(u, 4) + coe[5] * torch.pow(u, 5) \
                    + self.beta * (coe[6] + coe[7] * u + coe[8] * torch.pow(u, 2) + coe[9] * torch.pow(u, 3) + coe[10] * torch.pow(u, 4) + coe[11] * torch.pow(u, 5))
                return f
            elif self.power_nb == 4:
                coe = [ 1.4971e-02, -1.3835e+00,  1.5636e+01, -2.4473e+01,  1.1256e+01,
                        2.9137e-02, -8.3863e-01,  1.7329e+00, -7.5692e-01, -1.8362e-01]
                f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) + coe[3] * torch.pow(u, 3) + coe[4] * torch.pow(u, 4) \
                    + self.beta * (coe[5] + coe[6] * u + coe[7] * torch.pow(u, 2) + coe[8] * torch.pow(u, 3) + coe[9] * torch.pow(u, 4))
                return f
            elif self.power_nb == 3:
                coe = [-0.1428,  1.8134,  1.1932, -1.9714,  0.0316, -0.8895,  1.9653, -1.1219]
                f = coe[0] + coe[1] * u + coe[2] * torch.pow(u, 2) + coe[3] * torch.pow(u, 3) \
                    + self.beta * (coe[4] + coe[5] * u + coe[6] * torch.pow(u, 2) + coe[7] * torch.pow(u, 3))
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

    # def f_real(self, u):
    #     if is_real:
    #         f = (torch.pow(u, 2) * (1 - self.beta * torch.pow(1-u, 4))) / (torch.pow(u, 2) + 0.5 * torch.pow(1-u, 4))
    #         return f
    #     else:
    #         if self.power_nb == 8:
    #             molecular_coe = [ 4.5606e-04, -1.0869e-01,  1.0736e+00,  1.7163e+00, -1.1276e+00,
    #                               -3.5798e-01,  7.2708e+00,  1.4024e+01,  4.1288e+00, -8.6597e-06,
    #                               -2.1302e-02, -1.9570e-01, -6.8458e-01,  2.1531e+00,  5.8934e-01,
    #                               -3.0370e+00, -2.0360e+00,  4.0298e+00]
    #             denominator_coe = [ 2.1843e-01, -2.0391e-01,  6.7025e-01,  5.6516e-01,  1.0551e+00,
    #                                 2.7940e+00,  4.1274e+00,  6.0926e+00,  1.1397e+01, -8.2792e-03,
    #                                 1.0715e-01, -3.0087e-01, -1.9334e-01,  9.1613e-01,  5.3202e-01,
    #                                 -1.4133e+00, -2.2095e+00,  3.3439e+00]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) + molecular_coe[6] * torch.pow(u, 6) + molecular_coe[7] * torch.pow(u, 7) + molecular_coe[8] * torch.pow(u, 8) \
    #                    + self.beta * (molecular_coe[9] + molecular_coe[10] * u + molecular_coe[11] * torch.pow(u, 2) + molecular_coe[12] * torch.pow(u, 3) + molecular_coe[13] * torch.pow(u, 4) + molecular_coe[14] * torch.pow(u, 5) + molecular_coe[15] * torch.pow(u, 6) + molecular_coe[16] * torch.pow(u, 7) + molecular_coe[17] * torch.pow(u, 8))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) + denominator_coe[6] * torch.pow(u, 6) + denominator_coe[7] * torch.pow(u, 7) + denominator_coe[8] * torch.pow(u, 8) \
    #                    + self.beta * (denominator_coe[9] + denominator_coe[10] * u + denominator_coe[11] * torch.pow(u, 2) + denominator_coe[12] * torch.pow(u, 3) + denominator_coe[13] * torch.pow(u, 4) + denominator_coe[14] * torch.pow(u, 5) + denominator_coe[15] * torch.pow(u, 6) + denominator_coe[16] * torch.pow(u, 7) + denominator_coe[17] * torch.pow(u, 8))
    #             f = mole/deno
    #             return f
    #         elif self.power_nb == 7:
    #             molecular_coe = [-8.5269e-04, -3.9851e-02,  7.2015e-01,  4.6458e-02,  9.8653e-02,
    #                              7.5191e-01,  1.1766e+00,  7.4849e-01,  1.2460e-04, -2.8767e-02,
    #                              -1.0937e-01,  1.2293e-01,  1.4737e-01,  1.9967e-01, -1.4576e-01,
    #                              -1.7691e-01]
    #             denominator_coe = [1.7766e-01, -3.8585e-01,  5.6469e-01,  5.9339e-01,  3.0584e-01,
    #                                 2.1256e-01,  7.4027e-01,  1.2990e+00,  5.7440e-04,  2.4165e-02,
    #                                 -1.2227e-01,  6.7406e-02, -1.8451e-02,  3.1413e-01,  1.6092e-01,
    #                                 -4.1918e-01]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) + molecular_coe[6] * torch.pow(u, 6) + molecular_coe[7] * torch.pow(u, 7) \
    #                    + self.beta * (molecular_coe[8] + molecular_coe[9] * u + molecular_coe[10] * torch.pow(u, 2) + molecular_coe[11] * torch.pow(u, 3) + molecular_coe[12] * torch.pow(u, 4) + molecular_coe[13] * torch.pow(u, 5) + molecular_coe[14] * torch.pow(u, 6) + molecular_coe[15] * torch.pow(u, 7))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) + denominator_coe[6] * torch.pow(u, 6) + denominator_coe[7] * torch.pow(u, 7) \
    #                    + self.beta * (denominator_coe[8] + denominator_coe[9] * u + denominator_coe[10] * torch.pow(u, 2) + denominator_coe[11] * torch.pow(u, 3) + denominator_coe[12] * torch.pow(u, 4) + denominator_coe[13] * torch.pow(u, 5) + denominator_coe[14] * torch.pow(u, 6) + denominator_coe[15] * torch.pow(u, 7))
    #             f = mole/deno
    #             return f
    #         elif self.power_nb == 6:
    #             molecular_coe = [-5.0276e-04,  1.3425e-01, -1.3806e+00, -8.1740e-01, -8.5628e+00,
    #                              -4.0237e+00, -1.7831e+01,  4.8488e-05,  1.2095e-02,  3.5346e-01,
    #                              1.1697e+00, -3.9630e+00,  8.0325e+00, -1.0093e+01]
    #             denominator_coe = [-2.1892e-01, -3.3478e-01,  1.8038e-02, -3.1518e+00, -3.5086e+00,
    #                                -6.4429e+00, -1.8970e+01,  1.2175e-02, -1.5090e-01,  4.7286e-01,
    #                                -7.2084e-01,  1.4481e+00,  2.4891e+00, -7.9917e+00]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) + molecular_coe[6] * torch.pow(u, 6) \
    #                    + self.beta * (molecular_coe[7] + molecular_coe[8] * u + molecular_coe[9] * torch.pow(u, 2) + molecular_coe[10] * torch.pow(u, 3) + molecular_coe[11] * torch.pow(u, 4) + molecular_coe[12] * torch.pow(u, 5) + molecular_coe[13] * torch.pow(u, 6))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) + denominator_coe[6] * torch.pow(u, 6) \
    #                    + self.beta * (denominator_coe[7] + denominator_coe[8] * u + denominator_coe[9] * torch.pow(u, 2) + denominator_coe[10] * torch.pow(u, 3) + denominator_coe[11] * torch.pow(u, 4) + denominator_coe[12] * torch.pow(u, 5) + denominator_coe[13] * torch.pow(u, 6))
    #             f = mole/deno
    #             return f
    #         elif self.power_nb == 5:
    #             molecular_coe = [-3.7886e-04,  1.7722e-01, -2.1390e+00, -2.3398e+00,  2.9156e+00,
    #                              -5.6840e+00,  7.4598e-04,  2.0593e-02,  9.2035e-01, -1.9898e+00,
    #                              1.6017e+00, -1.4184e+00]
    #             denominator_coe = [-4.3133e-01,  4.2236e-01, -9.5366e-01, -1.4940e+00, -1.2897e+00,
    #                                -3.3011e+00, -1.3976e-03, -1.0103e-01,  4.5955e-01, -5.1161e-02,
    #                                -4.9796e-01, -6.6712e-01]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) \
    #                    + self.beta * (molecular_coe[6] + molecular_coe[7] * u + molecular_coe[8] * torch.pow(u, 2) + molecular_coe[9] * torch.pow(u, 3) + molecular_coe[10] * torch.pow(u, 4) + molecular_coe[11] * torch.pow(u, 5))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) \
    #                    + self.beta * (denominator_coe[6] + denominator_coe[7] * u + denominator_coe[8] * torch.pow(u, 2) + denominator_coe[9] * torch.pow(u, 3) + denominator_coe[10] * torch.pow(u, 4) + denominator_coe[11] * torch.pow(u, 5))
    #             f = mole/deno
    #             return f
    #         elif self.power_nb == 4:
    #             molecular_coe = [-8.4638e-04,  2.0199e-01, -2.3175e+00,  1.9913e+00, -9.5435e+00,
    #                              6.6544e-05,  2.5143e-03,  6.4379e-01, -3.5646e-01, -9.9890e-01]
    #             denominator_coe = [-0.2786, -0.0736, -0.4807,  0.8407, -9.6889,  0.0211, -0.2598,  0.7158,
    #                                0.1884, -1.3766]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) \
    #                    + self.beta * (molecular_coe[5] + molecular_coe[6] * u + molecular_coe[7] * torch.pow(u, 2) + molecular_coe[8] * torch.pow(u, 3) + molecular_coe[9] * torch.pow(u, 4))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) \
    #                    + self.beta * (denominator_coe[5] + denominator_coe[6] * u + denominator_coe[7] * torch.pow(u, 2) + denominator_coe[8] * torch.pow(u, 3) + denominator_coe[9] * torch.pow(u, 4))
    #             f = mole/deno
    #             return f
    #         elif self.power_nb == 3:
    #             molecular_coe = [-1.6937e-03, -1.6872e-02,  4.6222e-01,  1.3303e+00,  1.1976e-04,
    #                              -1.4622e-02, -2.8059e-01,  4.6949e-01]
    #             denominator_coe = [0.1713, -0.2219,  0.1518,  1.6842, -0.0082,  0.1406, -0.5663,  0.6089]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) \
    #                    + self.beta * (molecular_coe[4] + molecular_coe[5] * u + molecular_coe[6] * torch.pow(u, 2) + molecular_coe[7] * torch.pow(u, 3))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) \
    #                    + self.beta * (denominator_coe[4] + denominator_coe[5] * u + denominator_coe[6] * torch.pow(u, 2) + denominator_coe[7] * torch.pow(u, 3) )
    #             f = mole/deno
    #             return f

    # def f_real(self, u):
    #     if is_real:
    #         f = (torch.pow(u, 2) * (1 - self.beta * torch.pow(1-u, 4))) / (torch.pow(u, 2) + 0.5 * torch.pow(1-u, 4))
    #         return f
    #     else:
    #         if self.power_nb == 8:
    #             molecular_coe = [0.0130, -0.0650, -0.2105,  0.3674,  0.5193, -0.2884, -1.1686, -1.1012,
    #                              1.5198,  0.0103,  0.1247, -0.5446,  0.1668,  1.0928,  1.0850,  0.3070,
    #                              -0.7987, -0.1388]
    #             denominator_coe = [-0.0994,  0.3195, -0.1548,  0.1282,  0.0637, -0.0691,  0.0558,  0.0547,
    #                                -0.1635, -0.0173,  0.0482,  0.2313,  0.2386,  0.2640,  0.2238,  0.3241,
    #                                0.3282,  0.2853]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) + molecular_coe[6] * torch.pow(u, 6) + molecular_coe[7] * torch.pow(u, 7) + molecular_coe[8] * torch.pow(u, 8) \
    #                    + self.beta * (molecular_coe[9] + molecular_coe[10] * u + molecular_coe[11] * torch.pow(u, 2) + molecular_coe[12] * torch.pow(u, 3) + molecular_coe[13] * torch.pow(u, 4) + molecular_coe[14] * torch.pow(u, 5) + molecular_coe[15] * torch.pow(u, 6) + molecular_coe[16] * torch.pow(u, 7) + molecular_coe[17] * torch.pow(u, 8))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) + denominator_coe[6] * torch.pow(u, 6) + denominator_coe[7] * torch.pow(u, 7) + denominator_coe[8] * torch.pow(u, 8)\
    #                    + self.beta * (denominator_coe[9] + denominator_coe[10] * u + denominator_coe[11] * torch.pow(u, 2) + denominator_coe[12] * torch.pow(u, 3) + denominator_coe[13] * torch.pow(u, 4) + denominator_coe[14] * torch.pow(u, 5) + denominator_coe[15] * torch.pow(u, 6) + denominator_coe[16] * torch.pow(u, 7) + denominator_coe[17] * torch.pow(u, 8))
    #             f = mole/deno
    #             return f
    #         elif self.power_nb == 7:
    #             molecular_coe = [2.0186e-03, -2.7385e-01,  6.2890e-01,  8.1496e-01,  4.8249e-01,
    #                              2.6319e-01,  4.2647e-01,  3.6353e-01,  1.7418e-03, -2.7153e-01,
    #                              7.1917e-01,  5.8939e-01,  4.2922e-01,  4.8542e-01,  7.7713e-01,
    #                              2.5290e+00]
    #             denominator_coe = [0.1957, -0.0662, -0.2097,  0.3567,  1.1306,  1.2071,  0.5605, -0.4939,
    #                                0.2221, -0.1961,  0.0075,  0.4640,  0.5360,  1.2053,  1.5773,  1.4108]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) + molecular_coe[6] * torch.pow(u, 6) + molecular_coe[7] * torch.pow(u, 7) \
    #                    + self.beta * (molecular_coe[8] + molecular_coe[9] * u + molecular_coe[10] * torch.pow(u, 2) + molecular_coe[11] * torch.pow(u, 3) + molecular_coe[12] * torch.pow(u, 4) + molecular_coe[13] * torch.pow(u, 5) + molecular_coe[14] * torch.pow(u, 6) + molecular_coe[15] * torch.pow(u, 7))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) + denominator_coe[6] * torch.pow(u, 6) + denominator_coe[7] * torch.pow(u, 7) \
    #                    + self.beta * (denominator_coe[8] + denominator_coe[9] * u + denominator_coe[10] * torch.pow(u, 2) + denominator_coe[11] * torch.pow(u, 3) + denominator_coe[12] * torch.pow(u, 4) + denominator_coe[13] * torch.pow(u, 5) + denominator_coe[14] * torch.pow(u, 6) + denominator_coe[15] * torch.pow(u, 7))
    #             f = mole/deno
    #             return f
    #         elif self.power_nb == 6:
    #             molecular_coe = [-7.0699e-04,  1.0301e-01, -1.2928e+00, -1.0959e+00, -2.8249e-01,
    #                              -3.5936e+00, -7.7790e+00,  1.4742e-04,  3.7239e-02,  2.4613e-01,
    #                              4.2619e-01, -2.4086e+00,  3.4410e+00, -3.0852e+00]
    #             denominator_coe = [-2.6457e-01,  1.3283e-01, -1.2820e-01, -1.7147e+00, -1.6430e+00,
    #                                -2.1634e+00, -8.1376e+00,  2.6369e-03, -9.7678e-02,  4.7533e-01,
    #                                -6.4356e-01,  7.3440e-02,  1.4473e+00, -2.6043e+00]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) + molecular_coe[6] * torch.pow(u, 6)\
    #                    + self.beta * (molecular_coe[7] + molecular_coe[8] * u + molecular_coe[9] * torch.pow(u, 2) + molecular_coe[10] * torch.pow(u, 3) + molecular_coe[11] * torch.pow(u, 4) + molecular_coe[12] * torch.pow(u, 5) + molecular_coe[13] * torch.pow(u, 6))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) + denominator_coe[6] * torch.pow(u, 6)\
    #                    + self.beta * (denominator_coe[7] + denominator_coe[8] * u + denominator_coe[9] * torch.pow(u, 2) + denominator_coe[10] * torch.pow(u, 3) + denominator_coe[11] * torch.pow(u, 4) + denominator_coe[12] * torch.pow(u, 5) + denominator_coe[13] * torch.pow(u, 6))
    #             f = mole/deno
    #             return f
    #         elif self.power_nb == 5:
    #             molecular_coe = [2.9257e-04, -6.0451e-02,  8.5192e-01, -1.1258e-01,  3.8009e-01,
    #                              1.9196e+00, -1.1714e-04, -2.1085e-02, -1.8447e-01,  2.6088e-01,
    #                              7.7523e-02,  9.9530e-02]
    #             denominator_coe = [1.7557e-01, -3.1120e-01,  4.8654e-01,  5.0932e-01,  4.0039e-01,
    #                                1.7099e+00, -1.6422e-03,  4.8316e-02, -1.8631e-01,  6.4508e-02,
    #                                1.8323e-01,  1.2594e-01]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) + molecular_coe[5] * torch.pow(u, 5) \
    #                    + self.beta * (molecular_coe[6] + molecular_coe[7] * u + molecular_coe[8] * torch.pow(u, 2) + molecular_coe[9] * torch.pow(u, 3) + molecular_coe[10] * torch.pow(u, 4) + molecular_coe[11] * torch.pow(u, 5))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) + denominator_coe[5] * torch.pow(u, 5) \
    #                    + self.beta * (denominator_coe[6] + denominator_coe[7] * u + denominator_coe[8] * torch.pow(u, 2) + denominator_coe[9] * torch.pow(u, 3) + denominator_coe[10] * torch.pow(u, 4) + denominator_coe[11] * torch.pow(u, 5))
    #             f = mole/deno
    #             return f
    #         elif self.power_nb == 4:
    #             molecular_coe = [1.4001e-03, -2.2150e-01,  3.1953e+00, -2.2674e+00,  4.6245e+00,
    #                              -6.2421e-05, -1.0880e-01, -3.2954e-01,  2.8313e-01,  9.3187e-01]
    #             denominator_coe = [6.1315e-01, -1.1811e+00,  1.4468e+00,  1.9004e+00,  2.5291e+00,
    #                                1.2177e-02,  2.6833e-04, -9.8689e-02, -6.5726e-01,  1.5263e+00]
    #             mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) + molecular_coe[4] * torch.pow(u, 4) \
    #                    + self.beta * (molecular_coe[5] + molecular_coe[6] * u + molecular_coe[7] * torch.pow(u, 2) + molecular_coe[8] * torch.pow(u, 3) + molecular_coe[9] * torch.pow(u, 4))
    #             deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) + denominator_coe[4] * torch.pow(u, 4) \
    #                    + self.beta * (denominator_coe[5] + denominator_coe[6] * u + denominator_coe[7] * torch.pow(u, 2) + denominator_coe[8] * torch.pow(u, 3) + denominator_coe[9] * torch.pow(u, 4))
    #             f = mole/deno
    #             return f
    #         # elif self.power_nb == 3:
    #         #     molecular_coe = []
    #         #     denominator_coe = []
    #         #     mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) + molecular_coe[3] * torch.pow(u, 3) \
    #         #            + self.beta * (molecular_coe[4] + molecular_coe[5] * u + molecular_coe[6] * torch.pow(u, 2) + molecular_coe[7] * torch.pow(u, 3))
    #         #     deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) + denominator_coe[3] * torch.pow(u, 3) \
    #         #            + self.beta * (denominator_coe[4] + denominator_coe[5] * u + denominator_coe[6] * torch.pow(u, 2) + denominator_coe[7] * torch.pow(u, 3) )
    #         #     f = mole/deno
    #         #     return f
    #         # elif self.power_nb == 2:
    #         #     molecular_coe = []
    #         #     denominator_coe = []
    #         #     mole = molecular_coe[0] + molecular_coe[1] * u + molecular_coe[2] * torch.pow(u, 2) \
    #         #            + self.beta * (molecular_coe[3] + molecular_coe[4] * u + molecular_coe[5] * torch.pow(u, 2))
    #         #     deno = denominator_coe[0] + denominator_coe[1] * u + denominator_coe[2] * torch.pow(u, 2) \
    #         #            + self.beta * (denominator_coe[3] + denominator_coe[4] * u + denominator_coe[5] * torch.pow(u, 2))
    #         #     f = mole/deno
    #         #     return f
    #         # elif self.power_nb == 1:
    #         #     molecular_coe = []
    #         #     denominator_coe = []
    #         #     mole = molecular_coe[0] + molecular_coe[1] * u \
    #         #            + self.beta * (molecular_coe[2] + molecular_coe[3] * u)
    #         #     deno = denominator_coe[0] + denominator_coe[1] * u \
    #         #            + self.beta * (denominator_coe[2] + denominator_coe[3] * u)
    #         #     f = mole/deno
    #         #     return f


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
    # u_0
    batch_size = 4
    u_0_np = np.zeros((batch_size, N), dtype=float)
    u_0_np[:1, 180:240] = 1.0
    u_0_np[1:2, 140:200] = 0.9
    u_0_np[2:3, 200:260] = 0.8
    u_0_np[3:4, 120:240] = 0.7
    u_0 = torch.from_numpy(u_0_np)
    u_0 = u_0.to(device)
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
    # [1, 2, 3, 4, 5, 6, 7, 8]
    beta_list = [-3, -1, 9]
    for power_nb in [3, 4, 5, 6, 7, 8]:
        for beta in beta_list:
            for is_real in [True, False]:
                if beta < 0:
                    beta_name = 'ne_' + str(np.abs(beta))
                else:
                    beta_name = str(beta)
                experiment_name = 'N_400_example_4_beta_' + beta_name + '_power_np_' + str(power_nb)
                save_dir = 'data_m/' + experiment_name + '/'
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                if is_real:
                    real_data_file = '/' + experiment_name + '_real_U' + '.npy'
                    generate_real_data(save_dir + real_data_file, beta, power_nb, is_real)
                else:
                    real_data_file = '/' + experiment_name + '_predict_U' + '.npy'
                    generate_real_data(save_dir + real_data_file, beta, power_nb, is_real)




