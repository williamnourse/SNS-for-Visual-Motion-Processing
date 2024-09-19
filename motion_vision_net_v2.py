from snstorch import modules as m
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from motion_data import ClipDataset
import matplotlib.pyplot as plt

class AdaptiveSubnetwork(nn.Module):
    def __init__(self, num, height, width, tau_fastest, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.device = device
        self.dtype = dtype
        self.generator = generator
        self.tau_fastest = tau_fastest
        self.shape = [num, height, width]
        self.params = nn.ParameterDict({
            'ratioTauF': nn.Parameter(torch.rand(num, dtype=dtype, generator=generator).to(device)),
            'ratioTauS': nn.Parameter(torch.rand(num, dtype=dtype, generator=generator).to(device)),
            'bias': nn.Parameter(torch.zeros(self.shape, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape, dtype=dtype).to(device), requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape, dtype=dtype).to(device), requires_grad=False),
            'alpha': nn.Parameter(torch.rand(num, dtype=dtype, generator=generator).to(device)),
            'reversal': nn.Parameter((torch.tensor([5.0], dtype=dtype)).to(device), requires_grad=False),
        })
        if params is not None:
            self.params.update(params)

        self.fast = m.NonSpikingLayer(self.shape, generator=generator, device=device, dtype=dtype)
        self.slow = m.NonSpikingLayer(self.shape, generator=generator, device=device, dtype=dtype)
        self.syn_fs = m.NonSpikingChemicalSynapseElementwise(generator=generator, device=device, dtype=dtype)
        self.syn_sf = m.NonSpikingChemicalSynapseElementwise(generator=generator, device=device, dtype=dtype)

        self.setup()

    def init(self, batch_size=None):
        if batch_size is None:
            state_fast = torch.zeros(self.shape, dtype=self.dtype, device=self.device)# + self.params['rest'].unsqueeze(1).unsqueeze(1).expand(self.shape)
            state_slow = torch.zeros(self.shape, dtype=self.dtype, device=self.device)# + self.params['rest'].unsqueeze(1).unsqueeze(1).expand(self.shape)
        else:
            batch_shape = self.shape.copy()
            batch_shape.insert(0,batch_size)
            state_fast = torch.zeros(batch_shape, dtype=self.dtype, device=self.device)# + self.params['rest'].unsqueeze(1).unsqueeze(1).expand(batch_shape)
            state_slow = torch.zeros(batch_shape, dtype=self.dtype, device=self.device)# + self.params['rest'].unsqueeze(1).unsqueeze(1).expand(batch_shape)
        return state_fast, state_slow

    def setup(self):
        params_nrn_fast = nn.ParameterDict({
            'tau': nn.Parameter(self.tau_fastest*self.params['ratioTauF'].unsqueeze(1).unsqueeze(1).expand(self.shape) + torch.zeros(self.shape, dtype=self.dtype).to(self.device)),
            'leak': self.params['leak'],
            'rest': self.params['rest'],
            'bias': self.params['bias'],
            'init': nn.Parameter(torch.zeros(self.shape, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.fast.params.update(params_nrn_fast)

        params_nrn_slow = nn.ParameterDict({
            'tau': nn.Parameter(self.tau_fastest*self.params['ratioTauS'].unsqueeze(1).unsqueeze(1).expand(self.shape) + torch.zeros(self.shape, dtype=self.dtype).to(self.device)),
            'leak': self.params['leak'],
            'rest': self.params['rest'],
            'bias': self.params['bias'],
            'init': nn.Parameter(torch.zeros(self.shape, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.slow.params.update(params_nrn_slow)

        g_fs = 1/(self.params['reversal']-1)
        params_syn_fs = nn.ParameterDict({
            'conductance': nn.Parameter(g_fs.to(self.device), requires_grad=False),
            'reversal': self.params['reversal']
        })
        self.syn_fs.params.update(params_syn_fs)
        self.syn_fs.setup()

        g_sf = ((self.params['alpha']-1)*(self.params['reversal']-1+self.params['alpha']))/(self.params['alpha']*self.params['reversal']*(-self.params['reversal']-self.params['alpha']))
        params_syn_sf = nn.ParameterDict({
            'conductance': nn.Parameter(g_sf.unsqueeze(1).unsqueeze(1).expand(self.shape).to(self.device), requires_grad=False),
            'reversal': nn.Parameter(-self.params['reversal'].to(self.device), requires_grad=False)
        })
        self.syn_sf.params.update(params_syn_sf)
        self.syn_sf.setup()

    def forward(self, x, state_fast, state_slow):
        syn_fs = self.syn_fs(state_fast, state_slow)
        syn_sf = self.syn_sf(state_slow, state_fast)

        state_fast = self.fast(x+syn_sf, state_fast)
        state_slow = self.slow(syn_fs, state_slow)
        return state_fast, state_slow

class VisionNetv2(nn.Module):
    def __init__(self, shape, num_lamina, num_medulla, num_lobula, field, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.generator = generator
        self.dt = 1000*(1/30)/13
        self.tau_fastest = self.dt/(6*self.dt)
        self.params = nn.ParameterDict({
            'tauRatioRetina': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'tauRatioLobula': nn.Parameter(torch.rand(num_lobula, dtype=dtype, generator=generator).to(device)),
        })
        if params is not None:
            self.params.update(params)
        self.shape = shape
        self.shape_lamina = [x - (field-1) for x in self.shape]
        self.shape_medulla = [x - (field-1) for x in self.shape_lamina]
        self.shape_lobula = [x-2 for x in self.shape_medulla]
        self.shape_lobula.insert(0, num_lobula)

        self.nrn_retina = m.NonSpikingLayer(shape, device=device, dtype=dtype, generator=generator)
        self.syn_retina_lamina = m.NonSpikingChemicalSynapseConv(1, num_lamina, field, device=device, dtype=dtype, generator=generator)
        self.nrn_lamina = AdaptiveSubnetwork(num_lamina, self.shape_lamina[0], self.shape_lamina[1], self.tau_fastest, device=device, dtype=dtype, generator=generator)
        self.syn_lamina_medulla = m.NonSpikingChemicalSynapseConv(num_lamina, num_medulla, field, device=device, dtype=dtype, generator=generator)
        self.nrn_medulla = AdaptiveSubnetwork(num_medulla, self.shape_medulla[0], self.shape_medulla[1],
                                              self.tau_fastest, device=device, dtype=dtype, generator=generator)
        self.syn_medulla_lobula = m.NonSpikingChemicalSynapseConv(num_medulla, num_lobula, 3, device=device, dtype=dtype, generator=generator)
        self.nrn_lobula = m.NonSpikingLayer(self.shape_lobula, device=device, dtype=dtype, generator=generator)

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(num_lobula, 1, 5),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(2)
        )

        self.setup()

    def init(self, batch_size=None):
        state_retina = self.nrn_retina.params['init']
        state_lamina_fast, state_lamina_slow = self.nrn_lamina.init(batch_size=batch_size)
        state_medulla_fast, state_medulla_slow = self.nrn_medulla.init(batch_size=batch_size)
        state_lobula = self.nrn_lobula.params['init']
        return [state_retina, state_lamina_fast, state_lamina_slow, state_medulla_fast, state_medulla_slow, state_lobula]

    def setup(self):
        self.syn_retina_lamina.setup()
        params_retina = nn.ParameterDict({
            'tau': nn.Parameter(self.params['tauRatioRetina']*self.tau_fastest + torch.zeros(self.shape, dtype=self.dtype).to(self.device)),
            'leak': nn.Parameter(torch.ones(self.shape, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(torch.zeros(self.shape, dtype=self.dtype).to(self.device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.nrn_retina.params.update(params_retina)
        self.nrn_lamina.setup()
        self.syn_lamina_medulla.setup()
        self.nrn_medulla.setup()
        self.syn_medulla_lobula.setup()
        params_lobula = nn.ParameterDict({
            'tau': nn.Parameter(
                self.params['tauRatioLobula'].unsqueeze(1).unsqueeze(1).expand(self.shape_lobula)*self.tau_fastest),
            'leak': nn.Parameter(torch.ones(self.shape_lobula, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_lobula, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(torch.zeros(self.shape_lobula, dtype=self.dtype).to(self.device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_lobula, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.nrn_lobula.params.update(params_lobula)

    def forward(self, x, states):
        state_retina, state_lamina_fast, state_lamina_slow, state_medulla_fast, state_medulla_slow, state_lobula = states
        # Synapses
        syn_retina_lamina = self.syn_retina_lamina(state_retina, state_lamina_fast)
        syn_lamina_medulla = self.syn_lamina_medulla(state_lamina_fast, state_medulla_fast)
        syn_medulla_lobula = self.syn_medulla_lobula(state_medulla_fast, state_lobula)

        # Neurons
        state_retina = self.nrn_retina(x, state_retina)
        state_lamina_fast, state_lamina_slow = self.nrn_lamina(syn_retina_lamina, state_lamina_fast, state_lamina_slow)
        state_medulla_fast, state_medulla_slow = self.nrn_medulla(syn_lamina_medulla, state_medulla_fast, state_medulla_slow)
        state_lobula = self.nrn_lobula(syn_medulla_lobula, state_lobula)

        if (torch.isnan(state_retina).any() or torch.isnan(state_lamina_fast).any() or torch.isnan(state_lamina_slow).any()
                or torch.isnan(state_medulla_fast).any() or torch.isnan(state_medulla_slow).any() or torch.isnan(state_lobula).any()):
            print('Uh oh')

        return [state_retina, state_lamina_fast, state_lamina_slow, state_medulla_fast, state_medulla_slow, state_lobula]

class NetHandler(nn.Module):
    def __init__(self, net, shape, num_lamina, num_medulla, num_lobula, field, **kwargs):
        super().__init__()
        self.net = net(shape, num_lamina, num_medulla, num_lobula, field, **kwargs)

    def init(self, batch_size=None, input=None):
        states = self.net.init(batch_size=batch_size)
        return states

    def setup(self):
        self.net.setup()

    def forward(self, X, states):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        # raw: batch_size X n_steps X n_rows X n_cols
        X = X.permute(1, 0, 2, 3)

        self.batch_size = X.size(1)
        self.n_steps = X.size(0)
        self.n_substeps = 13

        # rnn_out => n_steps, batch_size, n_neurons (hidden states for each time step)
        # self.hidden => 1, batch_size, n_neurons (final state from each rnn_out)
        # running_ccw = torch.zeros(self.batch_size, dtype=self.net.dtype, device=self.net.device)
        # running_cw = torch.zeros(self.batch_size, dtype=self.net.dtype, device=self.net.device)
        # step = 0
        # while step < 400:
        #     states = self.net(X[0, :, :, :], states)
        #     step += 1
        step = 0
        size = self.n_substeps*self.n_steps
        lobula = torch.zeros_like(states[-1])
        for i in range(self.n_steps):
            for j in range(self.n_substeps):
                states = self.net(X[i,:,:, :], states)
                lobula = lobula + states[-1]/size
                step += 1
            # running_ccw += states[-1][:,1]
            # running_cw += states[-1][:, 0]
            # print(ext+prev)

        # print(out)
        return self.net.decoder(lobula), states


# net = NetHandler(VisionNetv2,[24,64], 5, 10, 4, 5)
# states = net.net.init(batch_size=6)
# decoded = net(torch.zeros([6,30,24,64]), states)
# print(decoded.shape)

# dt = (1/30)/13*1000
# tau_fastest = dt/(6*dt)
# params = nn.ParameterDict({
#             'ratioTauF': 1.0,
#             'ratioTauS': 0.1,
#             'alpha': 0.75,
#         })
# net = AdaptiveSubnetwork(1,1,1, tau_fastest)
#
# state_fast, state_slow = net.init()
# num_steps = 100
# fast = torch.zeros(num_steps)
# slow = torch.zeros(num_steps)
# for i in range(num_steps):
#     state_fast, state_slow = net(1,state_fast,state_slow)
#     fast[i] = state_fast
#     slow[i] = state_slow
#
# plt.figure()
# plt.plot(fast.detach(), label='F')
# plt.plot(slow.detach(), label='S')
# plt.axhline(y=net.params['alpha'].detach(), label='a', color='black')
# plt.legend()
#
# plt.show()
