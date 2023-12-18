import numpy as np
import torch
import sympy
ISINSTALLMATLAB = True
try:
    import matlab
except ModuleNotFoundError:
    ISINSTALLMATLAB = False
    matlab = None

__all__ = ['poly', ]
class poly(torch.nn.Module):
    def __init__(self, hidden_layers, channel_num, channel_names=None, theta=0.1, normalization_weight=None):
        super(poly, self).__init__()
        self.hidden_layers = hidden_layers
        self.channel_num = channel_num
        if channel_names is None:
            channel_names = list('u'+str(i) for i in range(self.channel_num))
        self.channel_names = channel_names
        layer = []
        # elements
        # for k in range(hidden_layers):
        #     module = torch.nn.Linear(channel_num+k, 2).to(dtype=torch.float64)
        #     module.weight.data.fill_(0)
        #     module.bias.data.fill_(0)
        #     self.add_module('layer'+str(k), module)
        #     layer.append(self.__getattr__('layer'+str(k)))
        # molecular
        module = torch.nn.Linear(11, 1, bias=False).to(dtype=torch.float64)
        module.weight.data.fill_(0)
        # module.bias.data.fill_(0)
        self.add_module('molecular', module)
        layer.append(self.__getattr__('molecular'))
        # denominator
        # module = torch.nn.Linear(7, 1).to(dtype=torch.float64)
        # module.weight.data.fill_(0)
        # module.bias.data.fill_(0)
        # self.add_module('denominator', module)
        # layer.append(self.__getattr__('denominator'))

        self.layer = tuple(layer)

        nw = torch.ones(channel_num).to(dtype=torch.float64)
        if (not isinstance(normalization_weight, torch.Tensor)) and (not normalization_weight is None):
            normalization_weight = np.array(normalization_weight)
            normalization_weight = torch.from_numpy(normalization_weight).to(dtype=torch.float64)
            normalization_weight = normalization_weight.view(channel_num)
            nw = normalization_weight
        self.register_buffer('_nw', nw)
        self.theta = theta

    @property
    def channels(self):
        channels = sympy.symbols(self.channel_names)
        return channels
    def renormalize(self, nw):
        if (not isinstance(nw, torch.Tensor)) and (not nw is None):
            nw = np.array(nw)
            nw = torch.from_numpy(nw)
        nw1 = nw.view(self.channel_num)
        nw1 = nw1.to(self._nw)
        nw0 = self._nw
        scale = nw0/nw1
        self._nw.data = nw1
        for L in self.layer:
            L.weight.data[:,:self.channel_num] *= scale
        return None
    def _cast2numpy(self, layer):
        weight, bias = layer.weight.data.cpu().numpy(), \
                      layer.bias.data.cpu().numpy()
        return weight, bias
    def _cast2matsym(self, layer, eng):
        weight,bias = self._cast2numpy(layer)
        weight,bias = weight.tolist(),bias.tolist()
        weight,bias = matlab.double(weight),matlab.double(bias)
        eng.workspace['weight'],eng.workspace['bias'] = weight,bias
        eng.workspace['weight'] = eng.eval("sym(weight,'d')")
        eng.workspace['bias'] = eng.eval("sym(bias,'d')")
        return None
    def _cast2symbol(self, layer):
        weight,bias = self._cast2numpy(layer)
        weight,bias = sympy.Matrix(weight),sympy.Matrix(bias)
        return weight,bias
    def _sympychop(self, o, calprec):
        cdict = o.expand().as_coefficients_dict()
        o = 0
        for k,v in cdict.items():
            if abs(v)>0.1**calprec:
                o = o+k*v
        return o
    def _matsymchop(self, o, calprec, eng):
        eng.eval('[c,t] = coeffs('+o+');', nargout=0)
        eng.eval('c = double(c);', nargout=0)
        eng.eval('c(abs(c)<1e-'+calprec+') = 0;', nargout=0)
        eng.eval(o+" = sum(sym(c, 'd').*t);", nargout=0)
        return None

    def expression(self, calprec=6, eng=None, isexpand=True):
        if eng is None:
            channels = sympy.symbols(self.channel_names)
            for i in range(self.channel_num):
                channels[i] = self._nw[i].item()*channels[i]
            channels = sympy.Matrix([channels,])
            for k in range(self.hidden_layers):
                weight, bias = self._cast2symbol(self.layer[k])
                o = weight*channels.transpose()+bias
                if isexpand:
                    o[0] = self._sympychop(o[0], calprec)
                    o[1] = self._sympychop(o[1], calprec)
                channels = list(channels)+[o[0]*o[1],]
                channels = sympy.Matrix([channels,])
            # molecular
            weight, bias = self._cast2symbol(self.layer[-2])
            o_molecular = (weight*channels.transpose()+bias)[0]
            if isexpand:
                o_molecular = o_molecular.expand()
                o_molecular = self._sympychop(o_molecular, calprec)
            # denominator
            weight, bias = self._cast2symbol(self.layer[-1])
            o_denominator = (weight*channels.transpose()+bias)[0]
            if isexpand:
                o_denominator = o_denominator.expand()
                o_denominator = self._sympychop(o_denominator, calprec)
            return o_molecular, o_denominator
        else:
            calprec = str(calprec)
            eng.clear(nargout=0)
            eng.syms(self.channel_names, nargout=0)
            channels = ""
            for c in self.channel_names:
                channels = channels+" "+c
            eng.eval('syms'+channels,nargout=0)
            channels = "["+channels+"].'"
            eng.workspace['channels'] = eng.eval(channels)
            eng.workspace['nw'] = matlab.double(self._nw.data.cpu().numpy().tolist())
            eng.eval("channels = channels.*nw.';", nargout=0)
            for k in range(self.hidden_layers):
                self._cast2matsym(self.layer[k], eng)
                eng.eval("o = weight*channels+bias';", nargout=0)
                eng.eval('o = o(1)*o(2);', nargout=0)
                if isexpand:
                    eng.eval('o = expand(o);', nargout=0)
                    self._matsymchop('o', calprec, eng)
                eng.eval('channels = [channels;o];', nargout=0)
            self._cast2matsym(self.layer[-1],eng)
            eng.eval("o = weight*channels+bias';", nargout=0)
            if isexpand:
                eng.eval("o = expand(o);", nargout=0)
                self._matsymchop('o', calprec, eng)
            return eng.workspace['o']

    def coeffs(self, calprec=6, eng=None, o=None, iprint=0):
        if eng is None:
            if o is None:
                o_molecular, o_denominator = self.expression(calprec, eng=None, isexpand=True)
            # molecular
            cdict_molecular = o_molecular.as_coefficients_dict()
            t_molecular = np.array(list(cdict_molecular.keys()))
            c_molecular = np.array(list(cdict_molecular.values()), dtype=np.float64)
            I_molecular = np.abs(c_molecular).argsort()[::-1]
            t_molecular = list(t_molecular[I_molecular])
            c_molecular = c_molecular[I_molecular]
            # denominator
            cdict_denominator = o_denominator.as_coefficients_dict()
            t_denominator = np.array(list(cdict_denominator.keys()))
            c_denominator = np.array(list(cdict_denominator.values()), dtype=np.float64)
            I_denominator = np.abs(c_denominator).argsort()[::-1]
            t_denominator = list(t_denominator[I_denominator])
            c_denominator = c_denominator[I_denominator]
            if iprint > 0:
                print(o)
            return t_molecular, c_molecular, t_denominator, c_denominator
        else:
            if o is None:
                self.expression(calprec, eng=eng, isexpand=True)
            else:
                eng.workspace['o'] = eng.expand(o)
            eng.eval('[c,t] = coeffs(o);', nargout=0)
            eng.eval('c = double(c);', nargout=0)
            eng.eval("[~,I] = sort(abs(c), 'descend'); c = c(I); t = t(I);", nargout=0)
            eng.eval('m = cell(numel(t),1);', nargout=0)
            eng.eval('for i=1:numel(t) m(i) = {char(t(i))}; end', nargout=0)
            if iprint > 0:
                eng.eval('disp(o)', nargout=0)
            t = list(eng.workspace['m'])
            c = np.array(eng.workspace['c']).flatten()
            return t, c

    def symboleval(self,inputs,eng=None,o=None):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.data.cpu().numpy()
        if isinstance(inputs, np.ndarray):
            inputs = list(inputs)
        assert len(inputs) == len(self.channel_names)
        if eng is None:
            if o is None:
                o = self.expression()
            return o.subs(dict(zip(self.channels,inputs)))
        else:
            if o is None:
                o = self.expression(eng=eng)
            channels = "["
            for c in self.channel_names:
                channels = channels+" "+c
            channels = channels+"].'"
            eng.workspace['channels'] = eng.eval(channels)
            eng.workspace['tmp'] = o
            eng.workspace['tmpv'] = matlab.double(inputs)
            eng.eval("tmpresults = double(subs(tmp,channels.',tmpv));",nargout=0)
            return np.array(eng.workspace['tmpresults'])

    # # 调整成为分段函数
    def u1(self, u):
        u1 = torch.where(u > -0.1, 1.0, 0.0)
        u2 = torch.where(u <= 0.0, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u2(self, u):
        u1 = torch.where(u > 0.0, 1.0, 0.0)
        u2 = torch.where(u <= 0.1, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u3(self, u):
        u1 = torch.where(u > 0.1, 1.0, 0.0)
        u2 = torch.where(u <= 0.2, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u4(self, u):
        u1 = torch.where(u > 0.2, 1.0, 0.0)
        u2 = torch.where(u <= 0.3, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u5(self, u):
        u1 = torch.where(u > 0.3, 1.0, 0.0)
        u2 = torch.where(u <= 0.4, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u6(self, u):
        u1 = torch.where(u > 0.4, 1.0, 0.0)
        u2 = torch.where(u <= 0.5, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u7(self, u):
        u1 = torch.where(u > 0.5, 1.0, 0.0)
        u2 = torch.where(u <= 0.6, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u8(self, u):
        u1 = torch.where(u > 0.6, 1.0, 0.0)
        u2 = torch.where(u <= 0.7, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u9(self, u):
        u1 = torch.where(u > 0.7, 1.0, 0.0)
        u2 = torch.where(u <= 0.8, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u10(self, u):
        u1 = torch.where(u > 0.8, 1.0, 0.0)
        u2 = torch.where(u <= 0.9, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u11(self, u):
        u1 = torch.where(u > 0.9, 1.0, 0.0)
        u2 = torch.where(u <= 1.0, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u12(self, u):
        u1 = torch.where(u > 1.0, 1.0, 0.0)
        u2 = torch.where(u <= 1.1, 1.0, 0.0)
        res = u1.mul(u2)
        return res



    def psi_0_l(self, u):
        return 10 * (u + 0.1)

    def psi_0_r(self, u):
        return 10 * (0.1 - u)

    def psi_1_l(self, u):
        return 10 * (u - 0)

    def psi_1_r(self, u):
        return 10 * (0.2 - u)

    def psi_2_l(self, u):
        return 10 * (u - 0.1)

    def psi_2_r(self, u):
        return 10 * (0.3 - u)

    def psi_3_l(self, u):
        return 10 * (u - 0.2)

    def psi_3_r(self, u):
        return 10 * (0.4 - u)

    def psi_4_l(self, u):
        return 10 * (u - 0.3)

    def psi_4_r(self, u):
        return 10 * (0.5 - u)

    def psi_5_l(self, u):
        return 10 * (u - 0.4)

    def psi_5_r(self, u):
        return 10 * (0.6 - u)

    def psi_6_l(self, u):
        return 10 * (u - 0.5)

    def psi_6_r(self, u):
        return 10 * (0.7 - u)

    def psi_7_l(self, u):
        return 10 * (u - 0.6)

    def psi_7_r(self, u):
        return 10 * (0.8 - u)

    def psi_8_l(self, u):
        return 10 * (u - 0.7)

    def psi_8_r(self, u):
        return 10 * (0.9 - u)

    def psi_9_l(self, u):
        return 10 * (u - 0.8)

    def psi_9_r(self, u):
        return 10 * (1.0 - u)

    def psi_10_l(self, u):
        return 10 * (u - 0.9)

    def psi_10_r(self, u):
        return 10 * (1.1 - u)


    def forward(self, inputs):
        outputs = inputs*self._nw
        v1 = self.psi_0_l(outputs) * self.u1(outputs)
        v2 = self.psi_0_r(outputs) * self.u2(outputs)

        v3 = self.psi_1_l(outputs) * self.u2(outputs)
        v4 = self.psi_1_r(outputs) * self.u3(outputs)

        v5 = self.psi_2_l(outputs) * self.u3(outputs)
        v6 = self.psi_2_r(outputs) * self.u4(outputs)

        v7 = self.psi_3_l(outputs) * self.u4(outputs)
        v8 = self.psi_3_r(outputs) * self.u5(outputs)

        v9 = self.psi_4_l(outputs) * self.u5(outputs)
        v10 = self.psi_4_r(outputs) * self.u6(outputs)

        v11 = self.psi_5_l(outputs) * self.u6(outputs)
        v12 = self.psi_5_r(outputs) * self.u7(outputs)

        v13 = self.psi_6_l(outputs) * self.u7(outputs)
        v14 = self.psi_6_r(outputs) * self.u8(outputs)

        v15 = self.psi_7_l(outputs) * self.u8(outputs)
        v16 = self.psi_7_r(outputs) * self.u9(outputs)

        v17 = self.psi_8_l(outputs) * self.u9(outputs)
        v18 = self.psi_8_r(outputs) * self.u10(outputs)

        v19 = self.psi_9_l(outputs) * self.u10(outputs)
        v20 = self.psi_9_r(outputs) * self.u11(outputs)

        v21 = self.psi_10_l(outputs) * self.u11(outputs)
        v22 = self.psi_10_r(outputs) * self.u12(outputs)

        v_t_1 = torch.cat([v1, v3, v5, v7, v9, v11, v13, v15, v17, v19, v21], dim=-1)
        res_1 = self.layer[-1](v_t_1)
        v_t_2 = torch.cat([v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22], dim=-1)
        res_2 = self.layer[-1](v_t_2)
        res = res_1 + res_2

        # res = self.psi_0_l(outputs) * self.u1(outputs) + self.psi_0_r(outputs) * self.u2(outputs)\
        #       + self.psi_1_l(outputs) * self.u2(outputs) + self.psi_1_r(outputs) * self.u3(outputs) \
        #       + self.psi_2_l(outputs) * self.u3(outputs) + self.psi_2_r(outputs) * self.u4(outputs) \
        #       + self.psi_3_l(outputs) * self.u4(outputs) + self.psi_3_r(outputs) * self.u5(outputs) \
        #       + self.psi_4_l(outputs) * self.u5(outputs) + self.psi_4_r(outputs) * self.u6(outputs) \
        #       + self.psi_5_l(outputs) * self.u6(outputs) + self.psi_5_r(outputs) * self.u7(outputs)
        return res[...,0]

    # def forward(self, inputs):
    #     outputs = inputs*self._nw
    #     repeat_num = 1
    #     for k in range(self.hidden_layers):
    #         # id
    #         id_list = []
    #         for index in range(outputs.shape[2]):
    #             for j in range(repeat_num):
    #                 id_list.append(outputs[:, :, index:index+1])
    #         id = torch.cat(id_list, dim=-1)
    #         # multiply
    #         o = self.layer[k](outputs)
    #         outputs = torch.cat([id, o[..., :1] * o[..., 1:]], dim=-1)
    #     # 除法
    #     r1 = self.layer[-2](outputs)
    #     r2 = self.layer[-1](outputs)
    #     # if (torch.min(r2) < self.theta):
    #     #     print("min")
    #     #     print(torch.min(r2))
    #     zero = torch.zeros_like(r2)
    #     outputs = torch.where(r2 < self.theta, zero, torch.div(r1, r2))
    #     return outputs[...,0]


    # def forward(self, inputs):
    #     outputs = inputs*self._nw
    #     repeat_num = 1
    #     # for k in range(self.hidden_layers):
    #     #     # id
    #     #     id_list = []
    #     #     for index in range(outputs.shape[2]):
    #     #         for j in range(repeat_num):
    #     #             id_list.append(outputs[:, :, index:index+1])
    #     #     id = torch.cat(id_list, dim=-1)
    #     #     # multiply
    #     #     o = self.layer[k](outputs)
    #     #     outputs = torch.cat([id, o[..., :1] * o[..., 1:]], dim=-1)
    #     # 不含有u的项
    #
    #     # v_0_m = torch.ones_like(outputs)
    #     v_1_m = outputs
    #     v_2_m = outputs * outputs
    #     v_3_m = outputs * outputs * outputs
    #     v_4_m = outputs * outputs * outputs * outputs
    #     v_5_m = outputs * outputs * outputs * outputs * outputs
    #     v_6_m = outputs * outputs * outputs * outputs * outputs * outputs
    #     v_7_m = outputs * outputs * outputs * outputs * outputs * outputs * outputs
    #     # v_8_m = outputs * outputs * outputs * outputs * outputs * outputs * outputs * outputs
    #     outputs_m = torch.cat([v_1_m, v_2_m, v_3_m, v_4_m, v_5_m, v_6_m, v_7_m], dim=-1)
    #
    #     # v_0_d = torch.ones_like(outputs)
    #     v_1_d = outputs
    #     v_2_d = outputs * outputs
    #     v_3_d = outputs * outputs * outputs
    #     v_4_d = outputs * outputs * outputs * outputs
    #     v_5_d = outputs * outputs * outputs * outputs * outputs
    #     v_6_d = outputs * outputs * outputs * outputs * outputs * outputs
    #     v_7_d = outputs * outputs * outputs * outputs * outputs * outputs * outputs
    #     # v_8_d = outputs * outputs * outputs * outputs * outputs * outputs * outputs * outputs
    #     outputs_d = torch.cat([v_1_d, v_2_d, v_3_d, v_4_d, v_5_d, v_6_d, v_7_d], dim=-1)
    #     # 除法
    #     r1 = self.layer[-2](outputs_m)
    #     r2 = self.layer[-1](outputs_d)
    #     zero = torch.zeros_like(r2)
    #     outputs = torch.where(r2 < self.theta, zero, torch.div(r1, r2))
    #     return outputs[...,0]








