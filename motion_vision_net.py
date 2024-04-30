from snstorch import modules as m
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List
import torch.jit as jit
import math
from motion_data import ClipDataset

def __calc_cap_from_cutoff__(cutoff):
    cap = 1000/(2*np.pi*cutoff)
    return cap

def __calc_2d_field__(amp_rel, std_cen, std_sur, shape_field, reversal_ex, reversal_in, device):
    axis = torch.tensor(np.arange(-(5*(shape_field-1)/2), 5*((shape_field-1)/2+1), 5))
    coeff_center = 1/(std_cen*torch.sqrt(torch.tensor([2*torch.pi], device=device)))
    coeff_surround = amp_rel/(std_sur*torch.sqrt(torch.tensor([2*torch.pi], device=device)))
    reversal = torch.zeros([shape_field, shape_field])
    target = torch.zeros([shape_field, shape_field])
    for i in range(shape_field):
        for j in range(shape_field):
            target[i,j] = -1 * coeff_center * torch.exp(-(axis[i] ** 2 + axis[j] ** 2) / (2 * std_cen**2)) + coeff_surround * torch.exp(
                -(axis[i] ** 2 + axis[j] ** 2) / (2 * std_sur**2))

            if target[i,j] >= 0:
                reversal[i,j] = reversal_ex
            else:
                reversal[i,j] = reversal_in

    conductance = target/reversal
    return conductance, reversal, target


class SNSBandpass(nn.Module):
    def __init__(self, shape, params=None, device=None, dtype=torch.float32, generator=None):
        """
        Implement a Bandpass filter as the difference of two lowpass filters
        :param shape: Tuple or array showing the shape of inputs
        :param params: ParameterDict of all the model_toolbox parameters
        :param device: Operating device, either cpu or cuda
        :param dtype: Datatype for all tensors, default is torch.float32
        :param generator: Generator object to use for random generation
        """
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'input_tau': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'input_leak': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'input_rest': nn.Parameter(torch.zeros(shape, dtype=dtype).to(device)),
            'input_bias': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'input_init': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'fast_tau': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'fast_leak': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'fast_rest': nn.Parameter(torch.zeros(shape, dtype=dtype).to(device)),
            'fast_bias': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'fast_init': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'slow_tau': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'slow_leak': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'slow_rest': nn.Parameter(torch.zeros(shape, dtype=dtype).to(device)),
            'slow_bias': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'slow_init': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'output_tau': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'output_leak': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'output_rest': nn.Parameter(torch.zeros(shape, dtype=dtype).to(device)),
            'output_bias': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'output_init': nn.Parameter(torch.rand(shape, dtype=dtype, generator=generator).to(device)),
            'reversalIn': nn.Parameter((torch.tensor([-2.0], dtype=dtype)).to(device)),
            'reversalEx': nn.Parameter((torch.tensor([2.0], dtype=dtype)).to(device)),
        })
        if params is not None:
            self.params.update(params)
        self.shape = shape
        self.dtype = dtype
        self.device = device

        self.input = m.NonSpikingLayer(shape, device=device, dtype=dtype)

        self.syn_input_fast = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.fast = m.NonSpikingLayer(shape, device=device, dtype=dtype)

        self.syn_input_slow = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.slow = m.NonSpikingLayer(shape, device=device, dtype=dtype)

        self.syn_fast_output = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.syn_slow_output = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.output = m.NonSpikingLayer(shape, device=device, dtype=dtype)

        self.setup()

    def forward(self, x, state_input, state_fast, state_slow, state_output):
        input2fast = self.syn_input_fast(state_input, state_fast)
        input2slow = self.syn_input_slow(state_input,  state_slow)
        fast2out = self.syn_fast_output(state_fast, state_output)
        slow2out = self.syn_slow_output(state_slow, state_output)

        state_input = self.input(x, state_input)
        state_fast = self.fast(input2fast, state_fast)
        state_slow = self.slow(input2slow, state_slow)
        state_output = self.output(fast2out+slow2out, state_output)

        return state_input, state_fast, state_slow, state_output

    # @jit.export
    def init(self, batch_size=None, input=None, input_cond=None, input_rev=None):
        if batch_size is None:
            state_input =  torch.zeros(self.shape, dtype=self.dtype, device=self.device) + self.params['input_init']
            state_fast =   torch.zeros(self.shape, dtype=self.dtype, device=self.device) + self.params['fast_init']
            state_slow =   torch.zeros(self.shape, dtype=self.dtype, device=self.device) + self.params['slow_init']
            state_output = torch.zeros(self.shape, dtype=self.dtype, device=self.device) + self.params['output_init']
        else:
            batch_shape = self.shape.copy()
            batch_shape.insert(0,batch_size)
            state_input = torch.zeros(batch_shape, dtype=self.dtype, device=self.device) + self.params['input_init']
            state_fast = torch.zeros(batch_shape, dtype=self.dtype, device=self.device) + self.params['fast_init']
            state_slow = torch.zeros(batch_shape, dtype=self.dtype, device=self.device) + self.params['slow_init']
            state_output = torch.zeros(batch_shape, dtype=self.dtype, device=self.device) + self.params['output_init']
        if input is not None:
            state_input = (nn.functional.conv2d(input, weight=input_cond)*torch.sum(input_rev)+1)/(1+nn.functional.conv2d(input, weight=input_cond))
            state_fast = (self.g_in*state_input*self.params['reversalIn']+1)/(1+self.g_in*state_input)
            state_slow = (self.g_in * state_input * self.params['reversalIn']+1) / (1 + self.g_in * state_input)
            state_output = (self.g_bd * state_fast * self.params['reversalIn']+1 + self.g_cd * state_slow *
                            self.params['reversalEx']) / (1 + self.g_bd * state_fast * self.params['reversalIn'] +
                                                          self.g_cd * state_slow * self.params['reversalEx'])
        return state_input, state_fast, state_slow, state_output

    # @jit.export
    def setup(self):
        # Retina
        k = 1.0
        # k = 1.141306594552181
        activity_range = 1.0
        self.g_in = (-activity_range) / self.params['reversalIn']
        self.g_bd = (-k * activity_range) / (self.params['reversalIn'] + k * activity_range)
        self.g_cd = (self.g_bd * (self.params['reversalIn'] - activity_range)) / (activity_range - self.params['reversalEx'])
        params_input = nn.ParameterDict({
            'tau': self.params['input_tau'],
            'leak': self.params['input_leak'],
            'rest': self.params['input_rest'],
            'bias': self.params['input_bias'],
            'init': self.params['input_init']
        })
        self.input.params.update(params_input)

        params_input_syn = nn.ParameterDict({
            'conductance': self.g_in,
            'reversal': self.params['reversalIn']
        })

        # Fast
        self.syn_input_fast.params.update(params_input_syn)
        params_fast = nn.ParameterDict({
            'tau': self.params['fast_tau'],
            'leak': self.params['fast_leak'],
            'rest': self.params['fast_rest'],
            'bias': self.params['fast_bias'],
            'init': self.params['fast_init']
        })
        self.fast.params.update(params_fast)

        # Slow
        self.syn_input_slow.params.update(params_input_syn)
        params_slow = nn.ParameterDict({
            'tau': self.params['slow_tau'],
            'leak': self.params['slow_leak'],
            'rest': self.params['slow_rest'],
            'bias': self.params['slow_bias'],
            'init': self.params['slow_init']
        })
        self.slow.params.update(params_slow)

        # Output
        params_fast_syn_output = nn.ParameterDict({
            'conductance': self.g_bd,
            'reversal': self.params['reversalIn']
        })
        self.syn_fast_output.params.update(params_fast_syn_output)
        params_slow_syn_output = nn.ParameterDict({
            'conductance': self.g_cd,
            'reversal': self.params['reversalEx']
        })
        self.syn_slow_output.params.update(params_slow_syn_output)
        params_output = nn.ParameterDict({
            'tau': self.params['output_tau'],
            'leak': self.params['output_leak'],
            'rest': self.params['output_rest'],
            'bias': self.params['output_bias'],
            'init': self.params['output_init']
        })
        self.output.params.update(params_output)


class VisionNet_1F(nn.Module):
    def __init__(self, dt, shape_input, shape_field, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        GENERAL
        """
        if device is None:
            device = 'cpu'
        self.device = device
        self.dtype = dtype
        self.shape_input = shape_input
        self.shape_field = shape_field
        self.shape_post_conv = [x - (shape_field-1) for x in self.shape_input]
        shape_emd = [x - 2 for x in self.shape_post_conv]
        self.shape_emd = shape_emd
        shape_emd_flat = shape_emd[0]*shape_emd[1]
        self.dt = dt

        self.tau_fast = self.dt/(6*self.dt)

        self.params = nn.ParameterDict({
            'reversalEx': nn.Parameter(torch.tensor([5.0], dtype=dtype).to(device), requires_grad=False),
            'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device), requires_grad=False),
            'reversalMod': nn.Parameter(torch.tensor([0.0], dtype=dtype).to(device), requires_grad=False),
            'stdCenBO': nn.Parameter(5*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampRelBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBO': nn.Parameter(20*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenL': nn.Parameter(5*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampRelL': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurL': nn.Parameter(20*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenBF': nn.Parameter(5*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampRelBF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBF': nn.Parameter(20*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauBF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceLEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceBODO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauDO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDOSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceLEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceBFDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDFSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceEOOn': nn.Parameter(10*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceEFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauOnCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauOnCCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasDO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauOffCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauOffCCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceOnCW': nn.Parameter((1/shape_emd_flat)*torch.rand(shape_emd_flat, dtype=dtype, generator=generator).to(device)),
            'conductanceOnCCW': nn.Parameter((1/shape_emd_flat)*torch.rand(shape_emd_flat, dtype=dtype, generator=generator).to(device)),
            'conductanceOffCW': nn.Parameter((1/shape_emd_flat)*torch.rand(shape_emd_flat, dtype=dtype, generator=generator).to(device)),
            'conductanceOffCCW': nn.Parameter((1/shape_emd_flat)*torch.rand(shape_emd_flat, dtype=dtype, generator=generator).to(device)),
            'reversalSignOnCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)-0.5),
            'reversalSignOnCCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)-0.5),
            'reversalSignOffCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)-0.5),
            'reversalSignOffCCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)-0.5),
            'ratioTauHorizontal': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
        })
        if params is not None:
            self.params.update(params)

        nrn_input_params = nn.ParameterDict({
            'tau': nn.Parameter((self.tau_fast + torch.zeros(shape_input, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_input, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False)
        })
        self.input = m.NonSpikingLayer(shape_input, params=nrn_input_params, device=device, dtype=dtype)

        # L
        self.syn_input_lowpass = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, device=device, dtype=dtype, generator=generator)
        self.lowpass = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # HS Cells

        flat_shape_emd = shape_emd[0] * shape_emd[1]
        self.hc = m.NonSpikingLayer([2], device=device, dtype=dtype)  # 0: CW, 1: CCW
        self.syn_on_ccw = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2, device=self.device, dtype=self.dtype)
        self.syn_on_cw = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2, device=self.device, dtype=self.dtype)
        self.syn_off_ccw = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2, device=self.device, dtype=self.dtype)
        self.syn_off_cw = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2, device=self.device, dtype=self.dtype)
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ON PATHWAY
        """
        """Lamina"""
        # Bo
        self.syn_input_bandpass_on = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, device=device, dtype=dtype, generator=generator)
        self.bandpass_on = SNSBandpass(self.shape_post_conv, device=device, dtype=dtype)

        """Medulla"""
        # EO
        self.syn_lowpass_enhance_on = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.enhance_on = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # DO
        self.syn_bandpass_on_direct_on = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.direct_on = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # SO
        self.syn_direct_on_suppress_on = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.suppress_on = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        """Lobula"""
        self.syn_direct_on_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)

        # CCW
        self.syn_enhance_on_ccw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.syn_suppress_on_ccw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.ccw_on = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)

        # CW
        self.syn_enhance_on_cw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.syn_suppress_on_cw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.cw_on = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)

        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        OFF PATHWAY
        """
        """Lamina"""
        # Bf
        self.syn_input_bandpass_off = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, device=device, dtype=dtype, generator=generator)
        self.bandpass_off = SNSBandpass(self.shape_post_conv, device=device, dtype=dtype)

        """Medulla"""
        # EO
        self.syn_lowpass_enhance_off = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.enhance_off = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # DO
        self.syn_bandpass_off_direct_off = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype, activation=m.PiecewiseActivation(1,2))
        self.direct_off = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # SO
        self.syn_direct_off_suppress_off = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.suppress_off = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        """Lobula"""
        self.syn_direct_off_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)

        # CCW
        self.syn_enhance_off_ccw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.syn_suppress_off_ccw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.ccw_off = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)

        # CW
        self.syn_enhance_off_cw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.syn_suppress_off_cw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.cw_off = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)


        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        SETUP
        """

        self.setup()

    def forward(self, x, states):
        [state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_bf_input,
         state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on, state_suppress_on,
         state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on, state_ccw_off,
         state_cw_off, state_hc] = states
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        SYNAPTIC UPDATES
        """
        # Retina -> Lamina
        syn_input_bandpass_on = self.syn_input_bandpass_on(state_input, state_bo_input)
        syn_input_lowpass = self.syn_input_lowpass(state_input, state_lowpass)
        syn_input_bandpass_off = self.syn_input_bandpass_off(state_input, state_bf_input)
        # Lamina -> Medulla
        syn_lowpass_enhance_on = self.syn_lowpass_enhance_on(state_lowpass, state_enhance_on)
        syn_bandpass_on_direct_on = self.syn_bandpass_on_direct_on(state_bo_output, state_direct_on)
        syn_direct_on_suppress_on = self.syn_direct_on_suppress_on(state_direct_on, state_suppress_on)
        syn_lowpass_enhance_off = self.syn_lowpass_enhance_off(state_lowpass, state_enhance_off)
        syn_bandpass_off_direct_off = self.syn_bandpass_off_direct_off(state_bf_output, state_direct_off)
        syn_direct_off_suppress_off = self.syn_direct_off_suppress_off(state_direct_off, state_suppress_off)
        # Medulla -> Lobula
        syn_enhance_on_ccw_on = self.syn_enhance_on_ccw_on(state_enhance_on, state_ccw_on)
        syn_direct_on_ccw_on = self.syn_direct_on_on(state_direct_on, state_ccw_on)
        syn_suppress_on_ccw_on = self.syn_suppress_on_ccw_on(state_suppress_on, state_ccw_on)
        syn_enhance_on_cw_on = self.syn_enhance_on_cw_on(state_enhance_on, state_cw_on)
        syn_direct_on_cw_on = self.syn_direct_on_on(state_direct_on, state_cw_on)
        syn_suppress_on_cw_on = self.syn_suppress_on_cw_on(state_suppress_on, state_cw_on)
        syn_enhance_off_ccw_off = self.syn_enhance_off_ccw_off(state_enhance_off, state_ccw_off)
        syn_direct_off_ccw_off = self.syn_direct_off_off(state_direct_off, state_ccw_off)
        syn_suppress_off_ccw_off = self.syn_suppress_off_ccw_off(state_suppress_off, state_ccw_off)
        syn_enhance_off_cw_off = self.syn_enhance_off_cw_off(state_enhance_off, state_cw_off)
        syn_direct_off_cw_off = self.syn_direct_off_off(state_direct_off, state_cw_off)
        syn_suppress_off_cw_off = self.syn_suppress_off_cw_off(state_suppress_off, state_cw_off)
        # Lobula -> Lobula Plate
        syn_on_cw_hc = self.syn_on_cw(state_cw_on.flatten(start_dim=1, end_dim=-1), state_hc)
        syn_on_ccw_hc = self.syn_on_ccw(state_ccw_on.flatten(start_dim=1, end_dim=-1), state_hc)
        syn_off_cw_hc = self.syn_off_cw(state_cw_off.flatten(start_dim=1, end_dim=-1), state_hc)
        syn_off_ccw_hc = self.syn_off_ccw(state_ccw_off.flatten(start_dim=1, end_dim=-1), state_hc)

        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        NEURAL UPDATES
        """
        # Retina
        state_input = self.input(x.squeeze(), state_input)
        # Lamina
        state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on(
            syn_input_bandpass_on.squeeze(), state_bo_input, state_bo_fast, state_bo_slow, state_bo_output)
        state_lowpass = self.lowpass(torch.squeeze(syn_input_lowpass), state_lowpass)
        state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off(
            syn_input_bandpass_off.squeeze(), state_bf_input, state_bf_fast, state_bf_slow, state_bf_output)
        # Medulla
        state_enhance_on = self.enhance_on(syn_lowpass_enhance_on, state_enhance_on)
        state_direct_on = self.direct_on(syn_bandpass_on_direct_on, state_direct_on)
        state_suppress_on = self.suppress_on(syn_direct_on_suppress_on, state_suppress_on)
        state_enhance_off = self.enhance_off(syn_lowpass_enhance_off, state_enhance_off)
        state_direct_off = self.direct_off(syn_bandpass_off_direct_off, state_direct_off)
        state_suppress_off = self.suppress_off(syn_direct_off_suppress_off, state_suppress_off)
        # Lobula
        state_ccw_on = self.ccw_on(torch.squeeze(syn_enhance_on_ccw_on+syn_direct_on_ccw_on+syn_suppress_on_ccw_on), state_ccw_on)
        state_cw_on = self.cw_on(torch.squeeze(syn_enhance_on_cw_on+syn_direct_on_cw_on+syn_suppress_on_cw_on), state_cw_on)
        state_ccw_off = self.ccw_off(torch.squeeze(syn_enhance_off_ccw_off + syn_direct_off_ccw_off + syn_suppress_off_ccw_off),
                                   state_ccw_off)
        state_cw_off = self.cw_off(torch.squeeze(syn_enhance_off_cw_off + syn_direct_off_cw_off + syn_suppress_off_cw_off),
                                 state_cw_off)
        # Lobula Plate
        state_hc = self.hc(syn_on_cw_hc+syn_on_ccw_hc+syn_off_cw_hc+syn_off_ccw_hc, state_hc)

        return [state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
                state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
                state_ccw_off, state_cw_off, state_hc]

    def init(self, batch_size=None, input=None):
        """
        Get all initial states
        :return:
        """
        if batch_size is None:
            if input is not None:
                state_input = input
                state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on.init(
                    batch_size=batch_size,
                    input=state_input,
                    input_cond=self.syn_input_bandpass_on.params['conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3),
                    input_rev=self.syn_input_bandpass_on.params['reversal'])
                state_lowpass = (nn.functional.conv2d(state_input,
                                                      weight=self.syn_input_lowpass.params['conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3)) *
                                 torch.sum(self.syn_input_lowpass.params['reversal']) + 1) / (
                                            1 + nn.functional.conv2d(state_input, weight=self.syn_input_lowpass.params[
                                        'conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3)))
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off.init(
                    batch_size=batch_size,
                    input=state_input,
                    input_cond=
                    self.syn_input_bandpass_off.params[
                        'conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3),
                    input_rev=
                    self.syn_input_bandpass_off.params[
                        'reversal'])
                state_enhance_on = ((self.params['conductanceLEO'] * state_lowpass * self.params['reversalEx'] +
                                     self.params['biasEO']) /
                                    (1 + self.params['conductanceLEO'] * state_lowpass))
                state_direct_on = ((self.params['conductanceBODO'] * state_bo_output * self.params['reversalIn'] +
                                    self.params['biasDO']) /
                                   (1 + self.params['conductanceBODO'] * state_bo_output))
                state_suppress_on = ((self.params['conductanceDOSO'] * state_direct_on * self.params['reversalEx'] +
                                      self.params['biasSO']) /
                                     (1 + self.params['conductanceDOSO'] * state_direct_on))
                state_enhance_off = ((self.params['conductanceLEF'] * state_lowpass * self.params['reversalEx'] +
                                      self.params['biasEF']) /
                                     (1 + self.params['conductanceLEF'] * state_lowpass))
                state_direct_off = ((self.params['conductanceBFDF'] * state_bf_output * self.params['reversalEx'] +
                                     self.params['biasDF']) /
                                    (1 + self.params['conductanceBFDF'] * state_bf_output))
                state_suppress_off = ((self.params['conductanceDFSF'] * state_direct_off * self.params['reversalEx'] +
                                       self.params['biasSF']) /
                                      (1 + self.params['conductanceDFSF'] * state_direct_off))
                state_ccw_on = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor([[[[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]]]]))*self.params['reversalMod'] +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor([[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]]))*self.params['reversalEx'] +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor([[[[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]]]]))*self.params['reversalIn'] +
                                self.params['biasOn']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]]]])) + 1)
                                )
                state_cw_on = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                    [[[[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]]]])) * self.params['reversalMod'] +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) * self.params[
                                     'reversalEx'] +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]]]])) * self.params[
                                     'reversalIn'] +
                                 self.params['biasOn']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]]]])) + 1)
                                )
                state_ccw_off = ((nn.functional.conv2d(state_enhance_off, weight=torch.tensor(
                    [[[[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]]]])) * self.params['reversalEx'] +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) * self.params[
                                     'reversalEx'] +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]]]])) * self.params[
                                     'reversalIn'] +
                                 self.params['biasOff']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]]]])) + 1)
                                )
                state_cw_off = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                    [[[[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]]]])) * self.params['reversalEx'] +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) * self.params[
                                    'reversalEx'] +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceSFOff'], 0, 0], [0, 0, 0]]]])) * self.params[
                                    'reversalIn'] +
                                self.params['biasOff']) /
                               (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                   [[[[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]]]])) +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceSFOff'], 0, 0], [0, 0, 0]]]])) + 1)
                               )
                state_hc = self.hc.params['init']
            else:
                state_input = self.input.params['init']
                state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on.init()
                state_lowpass = self.lowpass.params['init']
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off.init()
                state_enhance_on = self.enhance_on.params['init']
                state_direct_on = self.direct_on.params['init']
                state_suppress_on = self.suppress_on.params['init']
                state_enhance_off = self.enhance_off.params['init']
                state_direct_off = self.direct_off.params['init']
                state_suppress_off = self.suppress_off.params['init']
                state_ccw_on = self.ccw_on.params['init']
                state_cw_on = self.cw_on.params['init']
                state_ccw_off = self.ccw_off.params['init']
                state_cw_off = self.cw_off.params['init']
                state_hc = self.hc.params['init']
        else:
            if input is not None:
                state_input = input
                state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on.init(
                    batch_size=batch_size,
                    input=state_input,
                    input_cond=self.syn_input_bandpass_on.params['conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3),
                    input_rev=self.syn_input_bandpass_on.params['reversal'])
                state_lowpass = (nn.functional.conv2d(state_input,
                                                      weight=self.syn_input_lowpass.params['conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3)) *
                                 torch.sum(self.syn_input_lowpass.params['reversal']) + 1) / (
                                            1 + nn.functional.conv2d(state_input, weight=self.syn_input_lowpass.params[
                                        'conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3)))
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off.init(
                    batch_size=batch_size,
                    input=state_input,
                    input_cond=
                    self.syn_input_bandpass_off.params[
                        'conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3),
                    input_rev=
                    self.syn_input_bandpass_off.params[
                        'reversal'])
                state_enhance_on = ((self.params['conductanceLEO'] * state_lowpass * self.params['reversalEx'] +
                                     self.params['biasEO']) /
                                    (1 + self.params['conductanceLEO'] * state_lowpass))
                state_direct_on = ((self.params['conductanceBODO'] * state_bo_output * self.params['reversalIn'] +
                                    self.params['biasDO']) /
                                   (1 + self.params['conductanceBODO'] * state_bo_output))
                state_suppress_on = ((self.params['conductanceDOSO'] * state_direct_on * self.params['reversalEx'] +
                                      self.params['biasSO']) /
                                     (1 + self.params['conductanceDOSO'] * state_direct_on))
                state_enhance_off = ((self.params['conductanceLEF'] * state_lowpass * self.params['reversalEx'] +
                                      self.params['biasEF']) /
                                     (1 + self.params['conductanceLEF'] * state_lowpass))
                state_direct_off = ((self.params['conductanceBFDF'] * state_bf_output * self.params['reversalEx'] +
                                     self.params['biasDF']) /
                                    (1 + self.params['conductanceBFDF'] * state_bf_output))
                state_suppress_off = ((self.params['conductanceDFSF'] * state_direct_off * self.params['reversalEx'] +
                                       self.params['biasSF']) /
                                      (1 + self.params['conductanceDFSF'] * state_direct_off))
                state_ccw_on = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor([[[[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]]]]))*self.params['reversalMod'] +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor([[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]]))*self.params['reversalEx'] +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor([[[[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]]]]))*self.params['reversalIn'] +
                                self.params['biasOn']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]]]])) + 1)
                                )
                state_cw_on = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                    [[[[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]]]])) * self.params['reversalMod'] +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) * self.params[
                                     'reversalEx'] +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]]]])) * self.params[
                                     'reversalIn'] +
                                 self.params['biasOn']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]]]])) + 1)
                                )
                state_ccw_off = ((nn.functional.conv2d(state_enhance_off, weight=torch.tensor(
                    [[[[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]]]])) * self.params['reversalEx'] +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) * self.params[
                                     'reversalEx'] +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]]]])) * self.params[
                                     'reversalIn'] +
                                 self.params['biasOff']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]]]])) + 1)
                                )
                state_cw_off = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                    [[[[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]]]])) * self.params['reversalEx'] +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) * self.params[
                                    'reversalEx'] +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceSFOff'], 0, 0], [0, 0, 0]]]])) * self.params[
                                    'reversalIn'] +
                                self.params['biasOff']) /
                               (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                   [[[[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]]]])) +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceSFOff'], 0, 0], [0, 0, 0]]]])) + 1)
                               )
                state_hc = self.hc.params['init']
            else:
                batch_shape_input = self.shape_input.copy()
                batch_shape_input.insert(0,batch_size)
                batch_shape_post_conv = self.shape_post_conv.copy()
                batch_shape_post_conv.insert(0,batch_size)
                batch_shape_emd = self.shape_emd.copy()
                batch_shape_emd.insert(0, batch_size)
                state_input = self.input.params['init'] + torch.zeros(batch_shape_input, dtype=self.dtype, device=self.device)
                state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on.init(batch_size=batch_size)
                state_lowpass = self.lowpass.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off.init(batch_size=batch_size)
                state_enhance_on = self.enhance_on.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_direct_on = self.direct_on.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_suppress_on = self.suppress_on.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_enhance_off = self.enhance_off.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_direct_off = self.direct_off.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_suppress_off = self.suppress_off.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_ccw_on = self.ccw_on.params['init'] + torch.zeros(batch_shape_emd, dtype=self.dtype, device=self.device)
                state_cw_on = self.cw_on.params['init'] + torch.zeros(batch_shape_emd, dtype=self.dtype, device=self.device)
                state_ccw_off = self.ccw_off.params['init'] + torch.zeros(batch_shape_emd, dtype=self.dtype, device=self.device)
                state_cw_off = self.cw_off.params['init'] + torch.zeros(batch_shape_emd, dtype=self.dtype, device=self.device)
                state_hc = self.hc.params['init'] + torch.zeros([batch_size,2], dtype=self.dtype, device=self.device)


        return [state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
                state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
                state_ccw_off, state_cw_off, state_hc]

    def setup(self):
        """Lamina"""
        # Bandpass On
        conductance, reversal, _ = __calc_2d_field__(self.params['ampRelBO'], self.params['stdCenBO'],
                                                     self.params['stdSurBO'], self.shape_field,
                                                     self.params['reversalEx'], self.params['reversalIn'], self.device)
        syn_in_bo_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance.to(self.device), requires_grad=False),
            'reversal': nn.Parameter(reversal.to(self.device), requires_grad=False)
        })
        self.syn_input_bandpass_on.params.update(syn_in_bo_params)
        self.syn_input_bandpass_on.setup()
        tau_bo_fast = self.tau_fast
        tau_bo_slow = self.params['ratioTauBO']*self.tau_fast
        nrn_bo_params = nn.ParameterDict({
            'input_tau': nn.Parameter((self.tau_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype,
                                                                   device=self.device)).to(self.device),
                                      requires_grad=False),
            'input_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'fast_tau': nn.Parameter(
                (tau_bo_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'fast_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_tau': nn.Parameter(
                (tau_bo_slow + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'slow_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'output_tau': nn.Parameter(
                (self.tau_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(
                    self.device),
                requires_grad=False),
            'output_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'reversalIn': nn.Parameter((self.params['reversalIn'].clone().detach()).to(self.device),
                                       requires_grad=False),
            'reversalEx': nn.Parameter((self.params['reversalEx'].clone().detach()).to(self.device),
                                       requires_grad=False),
        })
        self.bandpass_on.params.update(nrn_bo_params)
        self.bandpass_on.setup()

        # Lowpass
        conductance, reversal, _ = __calc_2d_field__(self.params['ampRelL'], self.params['stdCenL'],
                                                     self.params['stdSurL'], self.shape_field,
                                                     self.params['reversalEx'], self.params['reversalIn'], self.device)
        syn_in_l_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance.to(self.device), requires_grad=False),
            'reversal': nn.Parameter(reversal.to(self.device), requires_grad=False)
        })
        self.syn_input_lowpass.params.update(syn_in_l_params)
        self.syn_input_lowpass.setup()
        # tau_l = self.dt / __calc_cap_from_cutoff__(self.params['freqLO'].data)
        tau_l = self.tau_fast
        nrn_l_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_l + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.lowpass.params.update(nrn_l_params)

        # Bandpass Off
        conductance, reversal, _ = __calc_2d_field__(self.params['ampRelBF'], self.params['stdCenBF'],
                                                     self.params['stdSurBF'], self.shape_field,
                                                     self.params['reversalEx'], self.params['reversalIn'], self.device)
        syn_in_bf_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance.to(self.device), requires_grad=False),
            'reversal': nn.Parameter(reversal.to(self.device), requires_grad=False)
        })
        self.syn_input_bandpass_off.params.update(syn_in_bf_params)
        self.syn_input_bandpass_off.setup()
        tau_bf_fast = self.tau_fast
        tau_bf_slow = self.params['ratioTauBF']*self.tau_fast
        nrn_bf_params = nn.ParameterDict({
            'input_tau': nn.Parameter((self.tau_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype,
                                                                   device=self.device)).to(self.device),
                                      requires_grad=False),
            'input_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'fast_tau': nn.Parameter(
                (tau_bf_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'fast_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_tau': nn.Parameter(
                (tau_bf_slow + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'slow_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'output_tau': nn.Parameter(
                (self.tau_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(
                    self.device),
                requires_grad=False),
            'output_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'reversalIn': nn.Parameter((self.params['reversalIn'].clone().detach()).to(self.device),
                                       requires_grad=False),
            'reversalEx': nn.Parameter((self.params['reversalEx'].clone().detach()).to(self.device),
                                       requires_grad=False),
        })
        self.bandpass_off.params.update(nrn_bf_params)
        self.bandpass_off.setup()

        """Medulla"""
        # Enhance On
        syn_l_eo_params = nn.ParameterDict({
            'conductance': self.params['conductanceLEO'],
            'reversal': self.params['reversalEx']
        })
        self.syn_lowpass_enhance_on.params.update(syn_l_eo_params)
        self.syn_lowpass_enhance_on.setup()
        tau_eo = self.params['ratioTauEO']*self.tau_fast
        nrn_eo_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_eo + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype, device=self.device).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasEO'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.enhance_on.params.update(nrn_eo_params)

        # Direct On
        syn_bo_do_params = nn.ParameterDict({
            'conductance': self.params['conductanceBODO'],
            'reversal': self.params['reversalIn']
        })
        self.syn_bandpass_on_direct_on.params.update(syn_bo_do_params)
        tau_do = self.tau_fast*self.params['ratioTauDO']
        nrn_do_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_do + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            # 'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasDO'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.direct_on.params.update(nrn_do_params)

        # Suppress On
        syn_do_so_params = nn.ParameterDict({
            'conductance': self.params['conductanceDOSO'],
            'reversal': self.params['reversalEx']
        })
        self.syn_direct_on_suppress_on.params.update(syn_do_so_params)
        tau_so = self.tau_fast*self.params['ratioTauSO']
        nrn_so_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_so + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasSO'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.suppress_on.params.update(nrn_so_params)

        # Enhance Off
        syn_l_ef_params = nn.ParameterDict({
            'conductance': self.params['conductanceLEF'],
            'reversal': self.params['reversalEx']
        })
        self.syn_lowpass_enhance_off.params.update(syn_l_ef_params)
        tau_ef = self.params['ratioTauEF']*self.tau_fast
        nrn_ef_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_ef + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype, device=self.device).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasEF'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.enhance_off.params.update(nrn_ef_params)

        # Direct Off
        syn_bf_df_params = nn.ParameterDict({
            'conductance': self.params['conductanceBFDF'],
            'reversal': self.params['reversalEx']
        })
        self.syn_bandpass_off_direct_off.params.update(syn_bf_df_params)
        tau_df = self.tau_fast*self.params['ratioTauDF']
        nrn_df_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_df + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            # 'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasDF'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.direct_off.params.update(nrn_df_params)

        # Suppress Off
        syn_df_sf_params = nn.ParameterDict({
            'conductance': self.params['conductanceDFSF'],
            'reversal': self.params['reversalEx']
        })
        self.syn_direct_off_suppress_off.params.update(syn_df_sf_params)
        tau_sf = self.tau_fast*self.params['ratioTauSF']
        nrn_sf_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_sf + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasSF'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.suppress_off.params.update(nrn_sf_params)

        """Lobula"""
        syn_do_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['reversalEx'], 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_direct_on_on.params.update(syn_do_on_params)
        self.syn_direct_on_on.setup()
        syn_df_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['reversalEx'], 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_direct_off_off.params.update(syn_df_off_params)
        self.syn_direct_off_off.setup()

        # CCW On Neuron
        syn_eo_ccw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalMod'], 0, 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_enhance_on_ccw_on.params.update(syn_eo_ccw_on_params)
        self.syn_enhance_on_ccw_on.setup()
        syn_so_ccw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalIn']], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_suppress_on_ccw_on.params.update(syn_so_ccw_on_params)
        self.syn_suppress_on_ccw_on.setup()
        tau_on_ccw = self.params['ratioTauOnCCW'] * self.tau_fast
        nrn_ccw_on_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_on_ccw + torch.zeros(self.shape_emd, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOn'] + torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.ccw_on.params.update(nrn_ccw_on_params)

        # CW On Neuron
        syn_eo_cw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalMod']], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_enhance_on_cw_on.params.update(syn_eo_cw_on_params)
        self.syn_enhance_on_cw_on.setup()
        syn_so_cw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalIn'], 0, 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_suppress_on_cw_on.params.update(syn_so_cw_on_params)
        self.syn_suppress_on_cw_on.setup()
        tau_on_cw = self.params['ratioTauOnCW'] * self.tau_fast
        nrn_cw_on_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_on_cw + torch.zeros(self.shape_emd, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOn'] + torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.cw_on.params.update(nrn_cw_on_params)

        # CCW Off Neuron
        syn_ef_ccw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalEx'], 0, 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_enhance_off_ccw_off.params.update(syn_ef_ccw_off_params)
        self.syn_enhance_off_ccw_off.setup()
        syn_sf_ccw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalIn']], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_suppress_off_ccw_off.params.update(syn_sf_ccw_off_params)
        self.syn_suppress_off_ccw_off.setup()
        tau_off_ccw = self.params['ratioTauOffCCW'] * self.tau_fast
        nrn_ccw_off_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_off_ccw + torch.zeros(self.shape_emd, dtype=self.dtype, device=self.device)).to(
                    self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOff'] + torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.ccw_off.params.update(nrn_ccw_off_params)

        # CW Off Neuron
        syn_ef_cw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalEx']], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_enhance_off_cw_off.params.update(syn_ef_cw_off_params)
        self.syn_enhance_off_cw_off.setup()
        syn_sf_cw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceSFOff'], 0, 0],
                                                      [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalIn'], 0, 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_suppress_off_cw_off.params.update(syn_sf_cw_off_params)
        self.syn_suppress_off_cw_off.setup()
        tau_off_cw = self.params['ratioTauOffCW'] * self.tau_fast
        nrn_cw_off_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_off_cw + torch.zeros(self.shape_emd, dtype=self.dtype, device=self.device)).to(
                    self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOff'] + torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.cw_off.params.update(nrn_cw_off_params)

        """Lobula Plate"""
        flat_shape_emd = self.shape_emd[0] * self.shape_emd[1]
        tau_hc = self.params['ratioTauHorizontal'] * self.tau_fast
        nrn_hc_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_hc + torch.zeros([2], dtype=self.dtype, device=self.device)).to(self.device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones([2], dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros([2], dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(torch.zeros([2], dtype=self.dtype).to(self.device), requires_grad=False),
            'init': nn.Parameter(torch.zeros([2], dtype=self.dtype).to(self.device), requires_grad=False)
        })
        self.hc.params.update(nrn_hc_params)

        g_on_cw = torch.clamp(self.params['conductanceOnCW'], min=0.0)
        g_on_ccw = torch.clamp(self.params['conductanceOnCCW'], min=0.0)
        g_off_cw = torch.clamp(self.params['conductanceOffCW'], min=0.0)
        g_off_ccw = torch.clamp(self.params['conductanceOffCCW'], min=0.0)

        g_on_cw_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_on_cw
        g_on_ccw_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_on_ccw
        g_off_cw_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_off_cw
        g_off_ccw_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_off_ccw

        # g_ex_tensor[:,(int(self.shape_emd[1] / 2) - 3):(int(self.shape_emd[1] / 2) + 3)] = 0.0
        # g_in_tensor[:,(int(self.shape_emd[1] / 2) - 3):(int(self.shape_emd[1] / 2) + 3)] = 0.0
        # g_ex_tensor = g_ex_tensor.flatten()
        # g_in_tensor = g_in_tensor.flatten()
        reversal_on_cw = (torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) +
                          torch.clamp(10*torch.sign(self.params['reversalSignOnCW']), min=self.params['reversalIn'], max=self.params['reversalEx']))
        reversal_on_ccw = (torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) +
                           torch.clamp(10*torch.sign(self.params['reversalSignOnCCW']), min=self.params['reversalIn'], max=self.params['reversalEx']))
        reversal_off_cw = (torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) +
                           torch.clamp(10*torch.sign(self.params['reversalSignOffCW']), min=self.params['reversalIn'], max=self.params['reversalEx']))
        reversal_off_ccw = (torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) +
                            torch.clamp(10*torch.sign(self.params['reversalSignOffCCW']), min=self.params['reversalIn'], max=self.params['reversalEx']))


        # Horizontal Cells
        syn_on_cw_hc_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_on_cw_tensor, g_on_ccw_tensor)).to(self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_on_cw, reversal_on_ccw)).to(self.device),
                                     requires_grad=False)
        })
        syn_on_ccw_hc_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_on_ccw_tensor, g_on_cw_tensor)).to(self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_on_ccw, reversal_on_cw)).to(self.device),
                                     requires_grad=False)
        })
        syn_off_cw_hc_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_off_cw_tensor, g_off_ccw_tensor)).to(self.device),
                                        requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_off_cw, reversal_off_ccw)).to(self.device),
                                     requires_grad=False)
        })
        syn_off_ccw_hc_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_off_ccw_tensor, g_off_cw_tensor)).to(self.device),
                                        requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_off_ccw, reversal_off_cw)).to(self.device),
                                     requires_grad=False)
        })
        self.syn_on_cw.params.update(syn_on_cw_hc_params)
        self.syn_on_ccw.params.update(syn_on_ccw_hc_params)
        self.syn_off_cw.params.update(syn_off_cw_hc_params)
        self.syn_off_ccw.params.update(syn_off_ccw_hc_params)

class VisionNet_1F_FB(nn.Module):
    def __init__(self, dt, shape_input, shape_field, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        GENERAL
        """
        if device is None:
            device = 'cpu'
        self.device = device
        self.dtype = dtype
        self.shape_input = shape_input
        self.shape_field = shape_field
        self.shape_post_conv = [x - (shape_field-1) for x in self.shape_input]
        shape_emd = [x - 2 for x in self.shape_post_conv]
        self.shape_emd = shape_emd
        shape_emd_flat = shape_emd[0]*shape_emd[1]
        self.dt = dt

        self.tau_fast = self.dt/(6*self.dt)

        self.params = nn.ParameterDict({
            'reversalEx': nn.Parameter(torch.tensor([5.0], dtype=dtype).to(device), requires_grad=False),
            'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device), requires_grad=False),
            'reversalMod': nn.Parameter(torch.tensor([0.0], dtype=dtype).to(device), requires_grad=False),
            'stdCenBO': nn.Parameter(5*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampRelBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBO': nn.Parameter(20*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauBO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenL': nn.Parameter(5*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampRelL': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurL': nn.Parameter(20*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdCenBF': nn.Parameter(5*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ampRelBF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'stdSurBF': nn.Parameter(20*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauBF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceLEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceBODO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauDO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDOSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceLEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceBFDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDFSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceEOOn': nn.Parameter(10*torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSOOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceEFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceDFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSFOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceSFEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauOnCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauOnCCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasEO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasDO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasSO': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasOn': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauOffCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'ratioTauOffCCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasEF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasDF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasSF': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'biasOff': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceOnCW': nn.Parameter((1/shape_emd_flat)*torch.rand(shape_emd_flat, dtype=dtype, generator=generator).to(device)),
            'conductanceOnCCW': nn.Parameter((1/shape_emd_flat)*torch.rand(shape_emd_flat, dtype=dtype, generator=generator).to(device)),
            'conductanceOffCW': nn.Parameter((1/shape_emd_flat)*torch.rand(shape_emd_flat, dtype=dtype, generator=generator).to(device)),
            'conductanceOffCCW': nn.Parameter((1/shape_emd_flat)*torch.rand(shape_emd_flat, dtype=dtype, generator=generator).to(device)),
            'reversalSignOnCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)-0.5),
            'reversalSignOnCCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)-0.5),
            'reversalSignOffCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)-0.5),
            'reversalSignOffCCW': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)-0.5),
            'ratioTauHorizontal': nn.Parameter(torch.rand(1, dtype=dtype, generator=generator).to(device)),
        })
        if params is not None:
            self.params.update(params)

        nrn_input_params = nn.ParameterDict({
            'tau': nn.Parameter((self.tau_fast + torch.zeros(shape_input, dtype=dtype, device=device)).to(device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones(shape_input, dtype=dtype).to(device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False),
            'init': nn.Parameter(torch.zeros(shape_input, dtype=dtype).to(device), requires_grad=False)
        })
        self.input = m.NonSpikingLayer(shape_input, params=nrn_input_params, device=device, dtype=dtype)

        # L
        self.syn_input_lowpass = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, device=device, dtype=dtype, generator=generator)
        self.lowpass = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # HS Cells

        flat_shape_emd = shape_emd[0] * shape_emd[1]
        self.hc = m.NonSpikingLayer([2], device=device, dtype=dtype)  # 0: CW, 1: CCW
        self.syn_on_ccw = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2, device=self.device, dtype=self.dtype)
        self.syn_on_cw = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2, device=self.device, dtype=self.dtype)
        self.syn_off_ccw = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2, device=self.device, dtype=self.dtype)
        self.syn_off_cw = m.NonSpikingChemicalSynapseLinear(flat_shape_emd, 2, device=self.device, dtype=self.dtype)
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ON PATHWAY
        """
        """Lamina"""
        # Bo
        self.syn_input_bandpass_on = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, device=device, dtype=dtype, generator=generator)
        self.bandpass_on = SNSBandpass(self.shape_post_conv, device=device, dtype=dtype)

        """Medulla"""
        # EO
        self.syn_lowpass_enhance_on = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.enhance_on = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # DO
        self.syn_bandpass_on_direct_on = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.direct_on = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # SO
        self.syn_direct_on_suppress_on = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.suppress_on = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        """Lobula"""
        self.syn_direct_on_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)

        # CCW
        self.syn_enhance_on_ccw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.syn_suppress_on_ccw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.ccw_on = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)

        # CW
        self.syn_enhance_on_cw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.syn_suppress_on_cw_on = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.cw_on = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)

        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        OFF PATHWAY
        """
        """Lamina"""
        # Bf
        self.syn_input_bandpass_off = m.NonSpikingChemicalSynapseConv(1, 1, shape_field, device=device, dtype=dtype, generator=generator)
        self.bandpass_off = SNSBandpass(self.shape_post_conv, device=device, dtype=dtype)

        """Medulla"""
        # EF
        self.syn_lowpass_enhance_off = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.syn_suppress_off_enhance_off = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.enhance_off = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # DF
        self.syn_bandpass_off_direct_off = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype, activation=m.PiecewiseActivation(1,2))
        self.direct_off = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        # SF
        self.syn_direct_off_suppress_off = m.NonSpikingChemicalSynapseElementwise(device=device, dtype=dtype)
        self.suppress_off = m.NonSpikingLayer(self.shape_post_conv, device=device, dtype=dtype)

        """Lobula"""
        self.syn_direct_off_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)

        # CCW
        self.syn_enhance_off_ccw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.syn_suppress_off_ccw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.ccw_off = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)

        # CW
        self.syn_enhance_off_cw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.syn_suppress_off_cw_off = m.NonSpikingChemicalSynapseConv(1, 1, 3, conv_dim=2, device=device, dtype=dtype)
        self.cw_off = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)


        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        SETUP
        """

        self.setup()

    def forward(self, x, states):
        [state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_bf_input,
         state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on, state_suppress_on,
         state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on, state_ccw_off,
         state_cw_off, state_hc] = states
        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        SYNAPTIC UPDATES
        """
        # Retina -> Lamina
        syn_input_bandpass_on = self.syn_input_bandpass_on(state_input, state_bo_input)
        syn_input_lowpass = self.syn_input_lowpass(state_input, state_lowpass)
        syn_input_bandpass_off = self.syn_input_bandpass_off(state_input, state_bf_input)
        # Lamina -> Medulla
        syn_lowpass_enhance_on = self.syn_lowpass_enhance_on(state_lowpass, state_enhance_on)
        syn_bandpass_on_direct_on = self.syn_bandpass_on_direct_on(state_bo_output, state_direct_on)
        syn_direct_on_suppress_on = self.syn_direct_on_suppress_on(state_direct_on, state_suppress_on)
        syn_lowpass_enhance_off = self.syn_lowpass_enhance_off(state_lowpass, state_enhance_off)
        syn_bandpass_off_direct_off = self.syn_bandpass_off_direct_off(state_bf_output, state_direct_off)
        syn_direct_off_suppress_off = self.syn_direct_off_suppress_off(state_direct_off, state_suppress_off)
        # Medulla -> Lobula
        syn_enhance_on_ccw_on = self.syn_enhance_on_ccw_on(state_enhance_on, state_ccw_on)
        syn_direct_on_ccw_on = self.syn_direct_on_on(state_direct_on, state_ccw_on)
        syn_suppress_on_ccw_on = self.syn_suppress_on_ccw_on(state_suppress_on, state_ccw_on)
        syn_enhance_on_cw_on = self.syn_enhance_on_cw_on(state_enhance_on, state_cw_on)
        syn_direct_on_cw_on = self.syn_direct_on_on(state_direct_on, state_cw_on)
        syn_suppress_on_cw_on = self.syn_suppress_on_cw_on(state_suppress_on, state_cw_on)
        syn_enhance_off_ccw_off = self.syn_enhance_off_ccw_off(state_enhance_off, state_ccw_off)
        syn_direct_off_ccw_off = self.syn_direct_off_off(state_direct_off, state_ccw_off)
        syn_suppress_off_ccw_off = self.syn_suppress_off_ccw_off(state_suppress_off, state_ccw_off)
        syn_enhance_off_cw_off = self.syn_enhance_off_cw_off(state_enhance_off, state_cw_off)
        syn_direct_off_cw_off = self.syn_direct_off_off(state_direct_off, state_cw_off)
        syn_suppress_off_cw_off = self.syn_suppress_off_cw_off(state_suppress_off, state_cw_off)
        syn_suppress_off_enhance_off = self.syn_suppress_off_enhance_off(state_suppress_off, state_enhance_off)
        # Lobula -> Lobula Plate
        syn_on_cw_hc = self.syn_on_cw(state_cw_on.flatten(start_dim=1, end_dim=-1), state_hc)
        syn_on_ccw_hc = self.syn_on_ccw(state_ccw_on.flatten(start_dim=1, end_dim=-1), state_hc)
        syn_off_cw_hc = self.syn_off_cw(state_cw_off.flatten(start_dim=1, end_dim=-1), state_hc)
        syn_off_ccw_hc = self.syn_off_ccw(state_ccw_off.flatten(start_dim=1, end_dim=-1), state_hc)

        """
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        NEURAL UPDATES
        """
        # Retina
        state_input = self.input(x.squeeze(), state_input)
        # Lamina
        state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on(
            syn_input_bandpass_on.squeeze(), state_bo_input, state_bo_fast, state_bo_slow, state_bo_output)
        state_lowpass = self.lowpass(torch.squeeze(syn_input_lowpass), state_lowpass)
        state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off(
            syn_input_bandpass_off.squeeze(), state_bf_input, state_bf_fast, state_bf_slow, state_bf_output)
        # Medulla
        state_enhance_on = self.enhance_on(syn_lowpass_enhance_on, state_enhance_on)
        state_direct_on = self.direct_on(syn_bandpass_on_direct_on, state_direct_on)
        state_suppress_on = self.suppress_on(syn_direct_on_suppress_on, state_suppress_on)
        state_enhance_off = self.enhance_off(syn_lowpass_enhance_off+syn_suppress_off_enhance_off, state_enhance_off)
        state_direct_off = self.direct_off(syn_bandpass_off_direct_off, state_direct_off)
        state_suppress_off = self.suppress_off(syn_direct_off_suppress_off, state_suppress_off)
        # Lobula
        state_ccw_on = self.ccw_on(torch.squeeze(syn_enhance_on_ccw_on+syn_direct_on_ccw_on+syn_suppress_on_ccw_on), state_ccw_on)
        state_cw_on = self.cw_on(torch.squeeze(syn_enhance_on_cw_on+syn_direct_on_cw_on+syn_suppress_on_cw_on), state_cw_on)
        state_ccw_off = self.ccw_off(torch.squeeze(syn_enhance_off_ccw_off + syn_direct_off_ccw_off + syn_suppress_off_ccw_off),
                                   state_ccw_off)
        state_cw_off = self.cw_off(torch.squeeze(syn_enhance_off_cw_off + syn_direct_off_cw_off + syn_suppress_off_cw_off),
                                 state_cw_off)
        # Lobula Plate
        state_hc = self.hc(syn_on_cw_hc+syn_on_ccw_hc+syn_off_cw_hc+syn_off_ccw_hc, state_hc)

        return [state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
                state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
                state_ccw_off, state_cw_off, state_hc]

    def init(self, batch_size=None, input=None):
        """
        Get all initial states
        :return:
        """
        if batch_size is None:
            if input is not None:
                state_input = input
                state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on.init(
                    batch_size=batch_size,
                    input=state_input,
                    input_cond=self.syn_input_bandpass_on.params['conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3),
                    input_rev=self.syn_input_bandpass_on.params['reversal'])
                state_lowpass = (nn.functional.conv2d(state_input,
                                                      weight=self.syn_input_lowpass.params['conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3)) *
                                 torch.sum(self.syn_input_lowpass.params['reversal']) + 1) / (
                                            1 + nn.functional.conv2d(state_input, weight=self.syn_input_lowpass.params[
                                        'conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3)))
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off.init(
                    batch_size=batch_size,
                    input=state_input,
                    input_cond=
                    self.syn_input_bandpass_off.params[
                        'conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3),
                    input_rev=
                    self.syn_input_bandpass_off.params[
                        'reversal'])
                state_enhance_on = ((self.params['conductanceLEO'] * state_lowpass * self.params['reversalEx'] +
                                     self.params['biasEO']) /
                                    (1 + self.params['conductanceLEO'] * state_lowpass))
                state_direct_on = ((self.params['conductanceBODO'] * state_bo_output * self.params['reversalIn'] +
                                    self.params['biasDO']) /
                                   (1 + self.params['conductanceBODO'] * state_bo_output))
                state_suppress_on = ((self.params['conductanceDOSO'] * state_direct_on * self.params['reversalEx'] +
                                      self.params['biasSO']) /
                                     (1 + self.params['conductanceDOSO'] * state_direct_on))
                state_direct_off = ((self.params['conductanceBFDF'] * state_bf_output * self.params['reversalEx'] +
                                     self.params['biasDF']) /
                                    (1 + self.params['conductanceBFDF'] * state_bf_output))
                state_suppress_off = ((self.params['conductanceDFSF'] * state_direct_off * self.params['reversalEx'] +
                                       self.params['biasSF']) /
                                      (1 + self.params['conductanceDFSF'] * state_direct_off))
                state_enhance_off = ((self.params['conductanceLEF'] * state_lowpass * self.params['reversalEx'] +
                                      self.params['conductanceSFEF'] * state_suppress_off * self.params['reversalIn'] +
                                      self.params['biasEF']) /
                                     (1 + self.params['conductanceLEF'] * state_lowpass + self.params['conductanceSFEF'] * state_suppress_off))
                state_ccw_on = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor([[[[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]]]]))*self.params['reversalMod'] +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor([[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]]))*self.params['reversalEx'] +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor([[[[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]]]]))*self.params['reversalIn'] +
                                self.params['biasOn']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]]]])) + 1)
                                )
                state_cw_on = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                    [[[[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]]]])) * self.params['reversalMod'] +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) * self.params[
                                     'reversalEx'] +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]]]])) * self.params[
                                     'reversalIn'] +
                                 self.params['biasOn']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]]]])) + 1)
                                )
                state_ccw_off = ((nn.functional.conv2d(state_enhance_off, weight=torch.tensor(
                    [[[[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]]]])) * self.params['reversalEx'] +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) * self.params[
                                     'reversalEx'] +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]]]])) * self.params[
                                     'reversalIn'] +
                                 self.params['biasOff']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]]]])) + 1)
                                )
                state_cw_off = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                    [[[[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]]]])) * self.params['reversalEx'] +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) * self.params[
                                    'reversalEx'] +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceSFOff'], 0, 0], [0, 0, 0]]]])) * self.params[
                                    'reversalIn'] +
                                self.params['biasOff']) /
                               (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                   [[[[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]]]])) +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceSFOff'], 0, 0], [0, 0, 0]]]])) + 1)
                               )
                state_hc = self.hc.params['init']
            else:
                state_input = self.input.params['init']
                state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on.init()
                state_lowpass = self.lowpass.params['init']
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off.init()
                state_enhance_on = self.enhance_on.params['init']
                state_direct_on = self.direct_on.params['init']
                state_suppress_on = self.suppress_on.params['init']
                state_enhance_off = self.enhance_off.params['init']
                state_direct_off = self.direct_off.params['init']
                state_suppress_off = self.suppress_off.params['init']
                state_ccw_on = self.ccw_on.params['init']
                state_cw_on = self.cw_on.params['init']
                state_ccw_off = self.ccw_off.params['init']
                state_cw_off = self.cw_off.params['init']
                state_hc = self.hc.params['init']
        else:
            if input is not None:
                state_input = input
                state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on.init(
                    batch_size=batch_size,
                    input=state_input,
                    input_cond=self.syn_input_bandpass_on.params['conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3),
                    input_rev=self.syn_input_bandpass_on.params['reversal'])
                state_lowpass = (nn.functional.conv2d(state_input,
                                                      weight=self.syn_input_lowpass.params['conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3)) *
                                 torch.sum(self.syn_input_lowpass.params['reversal']) + 1) / (
                                            1 + nn.functional.conv2d(state_input, weight=self.syn_input_lowpass.params[
                                        'conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3)))
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off.init(
                    batch_size=batch_size,
                    input=state_input,
                    input_cond=
                    self.syn_input_bandpass_off.params[
                        'conductance'].unsqueeze(0).unsqueeze(0).permute(1, 0, 2, 3),
                    input_rev=
                    self.syn_input_bandpass_off.params[
                        'reversal'])
                state_enhance_on = ((self.params['conductanceLEO'] * state_lowpass * self.params['reversalEx'] +
                                     self.params['biasEO']) /
                                    (1 + self.params['conductanceLEO'] * state_lowpass))
                state_direct_on = ((self.params['conductanceBODO'] * state_bo_output * self.params['reversalIn'] +
                                    self.params['biasDO']) /
                                   (1 + self.params['conductanceBODO'] * state_bo_output))
                state_suppress_on = ((self.params['conductanceDOSO'] * state_direct_on * self.params['reversalEx'] +
                                      self.params['biasSO']) /
                                     (1 + self.params['conductanceDOSO'] * state_direct_on))
                state_direct_off = ((self.params['conductanceBFDF'] * state_bf_output * self.params['reversalEx'] +
                                     self.params['biasDF']) /
                                    (1 + self.params['conductanceBFDF'] * state_bf_output))
                state_suppress_off = ((self.params['conductanceDFSF'] * state_direct_off * self.params['reversalEx'] +
                                       self.params['biasSF']) /
                                      (1 + self.params['conductanceDFSF'] * state_direct_off))
                state_enhance_off = ((self.params['conductanceLEF'] * state_lowpass * self.params['reversalEx'] +
                                      self.params['conductanceSFEF'] * state_suppress_off * self.params['reversalIn'] +
                                      self.params['biasEF']) /
                                     (1 + self.params['conductanceLEF'] * state_lowpass + self.params['conductanceSFEF'] * state_suppress_off))
                state_ccw_on = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor([[[[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]]]]))*self.params['reversalMod'] +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor([[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]]))*self.params['reversalEx'] +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor([[[[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]]]]))*self.params['reversalIn'] +
                                self.params['biasOn']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]]]])) + 1)
                                )
                state_cw_on = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                    [[[[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]]]])) * self.params['reversalMod'] +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) * self.params[
                                     'reversalEx'] +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]]]])) * self.params[
                                     'reversalIn'] +
                                 self.params['biasOn']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]]]])) + 1)
                                )
                state_ccw_off = ((nn.functional.conv2d(state_enhance_off, weight=torch.tensor(
                    [[[[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]]]])) * self.params['reversalEx'] +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) * self.params[
                                     'reversalEx'] +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]]]])) * self.params[
                                     'reversalIn'] +
                                 self.params['biasOff']) /
                                (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) +
                                 nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                     [[[[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]]]])) + 1)
                                )
                state_cw_off = ((nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                    [[[[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]]]])) * self.params['reversalEx'] +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) * self.params[
                                    'reversalEx'] +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceSFOff'], 0, 0], [0, 0, 0]]]])) * self.params[
                                    'reversalIn'] +
                                self.params['biasOff']) /
                               (nn.functional.conv2d(state_enhance_on, weight=torch.tensor(
                                   [[[[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]]]])) +
                                nn.functional.conv2d(state_direct_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]]]])) +
                                nn.functional.conv2d(state_suppress_on, weight=torch.tensor(
                                    [[[[0, 0, 0], [self.params['conductanceSFOff'], 0, 0], [0, 0, 0]]]])) + 1)
                               )
                state_hc = self.hc.params['init']
            else:
                batch_shape_input = self.shape_input.copy()
                batch_shape_input.insert(0,batch_size)
                batch_shape_post_conv = self.shape_post_conv.copy()
                batch_shape_post_conv.insert(0,batch_size)
                batch_shape_emd = self.shape_emd.copy()
                batch_shape_emd.insert(0, batch_size)
                state_input = self.input.params['init'] + torch.zeros(batch_shape_input, dtype=self.dtype, device=self.device)
                state_bo_input, state_bo_fast, state_bo_slow, state_bo_output = self.bandpass_on.init(batch_size=batch_size)
                state_lowpass = self.lowpass.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output = self.bandpass_off.init(batch_size=batch_size)
                state_enhance_on = self.enhance_on.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_direct_on = self.direct_on.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_suppress_on = self.suppress_on.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_enhance_off = self.enhance_off.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_direct_off = self.direct_off.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_suppress_off = self.suppress_off.params['init'] + torch.zeros(batch_shape_post_conv, dtype=self.dtype, device=self.device)
                state_ccw_on = self.ccw_on.params['init'] + torch.zeros(batch_shape_emd, dtype=self.dtype, device=self.device)
                state_cw_on = self.cw_on.params['init'] + torch.zeros(batch_shape_emd, dtype=self.dtype, device=self.device)
                state_ccw_off = self.ccw_off.params['init'] + torch.zeros(batch_shape_emd, dtype=self.dtype, device=self.device)
                state_cw_off = self.cw_off.params['init'] + torch.zeros(batch_shape_emd, dtype=self.dtype, device=self.device)
                state_hc = self.hc.params['init'] + torch.zeros([batch_size,2], dtype=self.dtype, device=self.device)


        return [state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
                state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
                state_ccw_off, state_cw_off, state_hc]

    def setup(self):
        """Lamina"""
        # Bandpass On
        conductance, reversal, _ = __calc_2d_field__(self.params['ampRelBO'], self.params['stdCenBO'],
                                                     self.params['stdSurBO'], self.shape_field,
                                                     self.params['reversalEx'], self.params['reversalIn'], self.device)
        syn_in_bo_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance.to(self.device), requires_grad=False),
            'reversal': nn.Parameter(reversal.to(self.device), requires_grad=False)
        })
        self.syn_input_bandpass_on.params.update(syn_in_bo_params)
        self.syn_input_bandpass_on.setup()
        tau_bo_fast = self.tau_fast
        tau_bo_slow = self.params['ratioTauBO']*self.tau_fast
        nrn_bo_params = nn.ParameterDict({
            'input_tau': nn.Parameter((self.tau_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype,
                                                                   device=self.device)).to(self.device),
                                      requires_grad=False),
            'input_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'fast_tau': nn.Parameter(
                (tau_bo_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'fast_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_tau': nn.Parameter(
                (tau_bo_slow + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'slow_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'output_tau': nn.Parameter(
                (self.tau_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(
                    self.device),
                requires_grad=False),
            'output_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'reversalIn': nn.Parameter((self.params['reversalIn'].clone().detach()).to(self.device),
                                       requires_grad=False),
            'reversalEx': nn.Parameter((self.params['reversalEx'].clone().detach()).to(self.device),
                                       requires_grad=False),
        })
        self.bandpass_on.params.update(nrn_bo_params)
        self.bandpass_on.setup()

        # Lowpass
        conductance, reversal, _ = __calc_2d_field__(self.params['ampRelL'], self.params['stdCenL'],
                                                     self.params['stdSurL'], self.shape_field,
                                                     self.params['reversalEx'], self.params['reversalIn'], self.device)
        syn_in_l_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance.to(self.device), requires_grad=False),
            'reversal': nn.Parameter(reversal.to(self.device), requires_grad=False)
        })
        self.syn_input_lowpass.params.update(syn_in_l_params)
        self.syn_input_lowpass.setup()
        # tau_l = self.dt / __calc_cap_from_cutoff__(self.params['freqLO'].data)
        tau_l = self.tau_fast
        nrn_l_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_l + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.lowpass.params.update(nrn_l_params)

        # Bandpass Off
        conductance, reversal, _ = __calc_2d_field__(self.params['ampRelBF'], self.params['stdCenBF'],
                                                     self.params['stdSurBF'], self.shape_field,
                                                     self.params['reversalEx'], self.params['reversalIn'], self.device)
        syn_in_bf_params = nn.ParameterDict({
            'conductance': nn.Parameter(conductance.to(self.device), requires_grad=False),
            'reversal': nn.Parameter(reversal.to(self.device), requires_grad=False)
        })
        self.syn_input_bandpass_off.params.update(syn_in_bf_params)
        self.syn_input_bandpass_off.setup()
        tau_bf_fast = self.tau_fast
        tau_bf_slow = self.params['ratioTauBF']*self.tau_fast
        nrn_bf_params = nn.ParameterDict({
            'input_tau': nn.Parameter((self.tau_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype,
                                                                   device=self.device)).to(self.device),
                                      requires_grad=False),
            'input_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'input_init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                       requires_grad=False),
            'fast_tau': nn.Parameter(
                (tau_bf_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'fast_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'fast_init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_tau': nn.Parameter(
                (tau_bf_slow + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'slow_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'slow_init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                      requires_grad=False),
            'output_tau': nn.Parameter(
                (self.tau_fast + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(
                    self.device),
                requires_grad=False),
            'output_leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_rest': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_bias': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'output_init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                        requires_grad=False),
            'reversalIn': nn.Parameter((self.params['reversalIn'].clone().detach()).to(self.device),
                                       requires_grad=False),
            'reversalEx': nn.Parameter((self.params['reversalEx'].clone().detach()).to(self.device),
                                       requires_grad=False),
        })
        self.bandpass_off.params.update(nrn_bf_params)
        self.bandpass_off.setup()

        """Medulla"""
        # Enhance On
        syn_l_eo_params = nn.ParameterDict({
            'conductance': self.params['conductanceLEO'],
            'reversal': self.params['reversalEx']
        })
        self.syn_lowpass_enhance_on.params.update(syn_l_eo_params)
        self.syn_lowpass_enhance_on.setup()
        tau_eo = self.params['ratioTauEO']*self.tau_fast
        nrn_eo_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_eo + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype, device=self.device).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasEO'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.enhance_on.params.update(nrn_eo_params)

        # Direct On
        syn_bo_do_params = nn.ParameterDict({
            'conductance': self.params['conductanceBODO'],
            'reversal': self.params['reversalIn']
        })
        self.syn_bandpass_on_direct_on.params.update(syn_bo_do_params)
        tau_do = self.tau_fast*self.params['ratioTauDO']
        nrn_do_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_do + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            # 'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasDO'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.direct_on.params.update(nrn_do_params)

        # Suppress On
        syn_do_so_params = nn.ParameterDict({
            'conductance': self.params['conductanceDOSO'],
            'reversal': self.params['reversalEx']
        })
        self.syn_direct_on_suppress_on.params.update(syn_do_so_params)
        tau_so = self.tau_fast*self.params['ratioTauSO']
        nrn_so_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_so + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasSO'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.suppress_on.params.update(nrn_so_params)

        # Enhance Off
        syn_l_ef_params = nn.ParameterDict({
            'conductance': self.params['conductanceLEF'],
            'reversal': self.params['reversalEx']
        })
        self.syn_lowpass_enhance_off.params.update(syn_l_ef_params)
        syn_sf_ef_params = nn.ParameterDict({
            'conductance': self.params['conductanceSFEF'],
            'reversal': self.params['reversalIn']
        })
        self.syn_suppress_off_enhance_off.params.update(syn_sf_ef_params)
        tau_ef = self.params['ratioTauEF']*self.tau_fast
        nrn_ef_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_ef + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype, device=self.device).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasEF'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.enhance_off.params.update(nrn_ef_params)

        # Direct Off
        syn_bf_df_params = nn.ParameterDict({
            'conductance': self.params['conductanceBFDF'],
            'reversal': self.params['reversalEx']
        })
        self.syn_bandpass_off_direct_off.params.update(syn_bf_df_params)
        tau_df = self.tau_fast*self.params['ratioTauDF']
        nrn_df_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_df + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            # 'bias': nn.Parameter(torch.zeros(shape_post_conv, dtype=dtype).to(device), requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasDF'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.direct_off.params.update(nrn_df_params)

        # Suppress Off
        syn_df_sf_params = nn.ParameterDict({
            'conductance': self.params['conductanceDFSF'],
            'reversal': self.params['reversalEx']
        })
        self.syn_direct_off_suppress_off.params.update(syn_df_sf_params)
        tau_sf = self.tau_fast*self.params['ratioTauSF']
        nrn_sf_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_sf + torch.zeros(self.shape_post_conv, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'bias': nn.Parameter(
                self.params['biasSF'] + torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_post_conv, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
        })
        self.suppress_off.params.update(nrn_sf_params)

        """Lobula"""
        syn_do_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['conductanceDOOn'], 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['reversalEx'], 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_direct_on_on.params.update(syn_do_on_params)
        self.syn_direct_on_on.setup()
        syn_df_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['conductanceDFOff'], 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, self.params['reversalEx'], 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_direct_off_off.params.update(syn_df_off_params)
        self.syn_direct_off_off.setup()

        # CCW On Neuron
        syn_eo_ccw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceEOOn'], 0, 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalMod'], 0, 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_enhance_on_ccw_on.params.update(syn_eo_ccw_on_params)
        self.syn_enhance_on_ccw_on.setup()
        syn_so_ccw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceSOOn']], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalIn']], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_suppress_on_ccw_on.params.update(syn_so_ccw_on_params)
        self.syn_suppress_on_ccw_on.setup()
        tau_on_ccw = self.params['ratioTauOnCCW'] * self.tau_fast
        nrn_ccw_on_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_on_ccw + torch.zeros(self.shape_emd, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOn'] + torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.ccw_on.params.update(nrn_ccw_on_params)

        # CW On Neuron
        syn_eo_cw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceEOOn']], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalMod']], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_enhance_on_cw_on.params.update(syn_eo_cw_on_params)
        self.syn_enhance_on_cw_on.setup()
        syn_so_cw_on_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceSOOn'], 0, 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalIn'], 0, 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_suppress_on_cw_on.params.update(syn_so_cw_on_params)
        self.syn_suppress_on_cw_on.setup()
        tau_on_cw = self.params['ratioTauOnCW'] * self.tau_fast
        nrn_cw_on_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_on_cw + torch.zeros(self.shape_emd, dtype=self.dtype, device=self.device)).to(self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOn'] + torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.cw_on.params.update(nrn_cw_on_params)

        # CCW Off Neuron
        syn_ef_ccw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceEFOff'], 0, 0], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalEx'], 0, 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_enhance_off_ccw_off.params.update(syn_ef_ccw_off_params)
        self.syn_enhance_off_ccw_off.setup()
        syn_sf_ccw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceSFOff']], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalIn']], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_suppress_off_ccw_off.params.update(syn_sf_ccw_off_params)
        self.syn_suppress_off_ccw_off.setup()
        tau_off_ccw = self.params['ratioTauOffCCW'] * self.tau_fast
        nrn_ccw_off_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_off_ccw + torch.zeros(self.shape_emd, dtype=self.dtype, device=self.device)).to(
                    self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOff'] + torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.ccw_off.params.update(nrn_ccw_off_params)

        # CW Off Neuron
        syn_ef_cw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['conductanceEFOff']], [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [0, 0, self.params['reversalEx']], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_enhance_off_cw_off.params.update(syn_ef_cw_off_params)
        self.syn_enhance_off_cw_off.setup()
        syn_sf_cw_off_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['conductanceSFOff'], 0, 0],
                                                      [0, 0, 0]],
                                                     dtype=self.dtype, device=self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.tensor([[0, 0, 0], [self.params['reversalIn'], 0, 0], [0, 0, 0]],
                                                  dtype=self.dtype, device=self.device), requires_grad=False),
        })
        self.syn_suppress_off_cw_off.params.update(syn_sf_cw_off_params)
        self.syn_suppress_off_cw_off.setup()
        tau_off_cw = self.params['ratioTauOffCW'] * self.tau_fast
        nrn_cw_off_params = nn.ParameterDict({
            'tau': nn.Parameter(
                (tau_off_cw + torch.zeros(self.shape_emd, dtype=self.dtype, device=self.device)).to(
                    self.device),
                requires_grad=False),
            'leak': nn.Parameter(torch.ones(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(self.params['biasOff'] + torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device),
                                 requires_grad=False),
            'init': nn.Parameter(torch.zeros(self.shape_emd, dtype=self.dtype).to(self.device), requires_grad=False),
        })
        self.cw_off.params.update(nrn_cw_off_params)

        """Lobula Plate"""
        flat_shape_emd = self.shape_emd[0] * self.shape_emd[1]
        tau_hc = self.params['ratioTauHorizontal'] * self.tau_fast
        nrn_hc_params = nn.ParameterDict({
            'tau': nn.Parameter((tau_hc + torch.zeros([2], dtype=self.dtype, device=self.device)).to(self.device),
                                requires_grad=False),
            'leak': nn.Parameter(torch.ones([2], dtype=self.dtype).to(self.device), requires_grad=False),
            'rest': nn.Parameter(torch.zeros([2], dtype=self.dtype).to(self.device), requires_grad=False),
            'bias': nn.Parameter(torch.zeros([2], dtype=self.dtype).to(self.device), requires_grad=False),
            'init': nn.Parameter(torch.zeros([2], dtype=self.dtype).to(self.device), requires_grad=False)
        })
        self.hc.params.update(nrn_hc_params)

        g_on_cw = torch.clamp(self.params['conductanceOnCW'], min=0.0)
        g_on_ccw = torch.clamp(self.params['conductanceOnCCW'], min=0.0)
        g_off_cw = torch.clamp(self.params['conductanceOffCW'], min=0.0)
        g_off_ccw = torch.clamp(self.params['conductanceOffCCW'], min=0.0)

        g_on_cw_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_on_cw
        g_on_ccw_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_on_ccw
        g_off_cw_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_off_cw
        g_off_ccw_tensor = torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) + g_off_ccw

        # g_ex_tensor[:,(int(self.shape_emd[1] / 2) - 3):(int(self.shape_emd[1] / 2) + 3)] = 0.0
        # g_in_tensor[:,(int(self.shape_emd[1] / 2) - 3):(int(self.shape_emd[1] / 2) + 3)] = 0.0
        # g_ex_tensor = g_ex_tensor.flatten()
        # g_in_tensor = g_in_tensor.flatten()
        reversal_on_cw = (torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) +
                          torch.clamp(10*torch.sign(self.params['reversalSignOnCW']), min=self.params['reversalIn'], max=self.params['reversalEx']))
        reversal_on_ccw = (torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) +
                           torch.clamp(10*torch.sign(self.params['reversalSignOnCCW']), min=self.params['reversalIn'], max=self.params['reversalEx']))
        reversal_off_cw = (torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) +
                           torch.clamp(10*torch.sign(self.params['reversalSignOffCW']), min=self.params['reversalIn'], max=self.params['reversalEx']))
        reversal_off_ccw = (torch.zeros(flat_shape_emd, dtype=self.dtype, device=self.device) +
                            torch.clamp(10*torch.sign(self.params['reversalSignOffCCW']), min=self.params['reversalIn'], max=self.params['reversalEx']))


        # Horizontal Cells
        syn_on_cw_hc_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_on_cw_tensor, g_on_ccw_tensor)).to(self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_on_cw, reversal_on_ccw)).to(self.device),
                                     requires_grad=False)
        })
        syn_on_ccw_hc_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_on_ccw_tensor, g_on_cw_tensor)).to(self.device), requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_on_ccw, reversal_on_cw)).to(self.device),
                                     requires_grad=False)
        })
        syn_off_cw_hc_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_off_cw_tensor, g_off_ccw_tensor)).to(self.device),
                                        requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_off_cw, reversal_off_ccw)).to(self.device),
                                     requires_grad=False)
        })
        syn_off_ccw_hc_params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.vstack((g_off_ccw_tensor, g_off_cw_tensor)).to(self.device),
                                        requires_grad=False),
            'reversal': nn.Parameter(torch.vstack((reversal_off_ccw, reversal_off_cw)).to(self.device),
                                     requires_grad=False)
        })
        self.syn_on_cw.params.update(syn_on_cw_hc_params)
        self.syn_on_ccw.params.update(syn_on_ccw_hc_params)
        self.syn_off_cw.params.update(syn_off_cw_hc_params)
        self.syn_off_ccw.params.update(syn_off_ccw_hc_params)

class NetHandler(nn.Module):
    def __init__(self, net, dt, shape_input, shape_field, **kwargs):
        super().__init__()
        self.net = net(dt, shape_input, shape_field, **kwargs)

    def init(self, batch_size=None, input=None):
        states = self.net.init(batch_size=batch_size, input=input)
        return states

    def setup(self):
        self.net.setup()

    def forward(self, X, states):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        # raw: batch_size X n_steps X n_rows X n_cols
        X = X.permute(1, 0, 2, 3)

        self.batch_size = X.size(1)
        self.n_steps = X.size(0)

        # rnn_out => n_steps, batch_size, n_neurons (hidden states for each time step)
        # self.hidden => 1, batch_size, n_neurons (final state from each rnn_out)
        # running_ccw = torch.zeros(self.batch_size, dtype=self.net.dtype, device=self.net.device)
        # running_cw = torch.zeros(self.batch_size, dtype=self.net.dtype, device=self.net.device)
        # step = 0
        # while step < 400:
        #     states = self.net(X[0, :, :, :], states)
        #     step += 1
        for i in range(self.n_steps):
            states = self.net(X[i,:,:, :], states)
            # running_ccw += states[-1][:,1]
            # running_cw += states[-1][:, 0]
            # print(ext+prev)

        # print(out)
        return torch.clamp(states[-1], min=0)
        # return running_ccw/i, running_cw/i  # batch_size X n_output

if __name__ == "__main__":
    img_size = [24,64]
    train = ClipDataset('/home/will/flywheel-rotation-dataset/FlyWheelTrain3s')
    sample_img, sample_label = train[0]
    print(train.__len__())
    print(sample_img.shape, sample_label)

    train_dataloader = DataLoader(train, batch_size=3, shuffle=True)
    imgs, labels = next(iter(train_dataloader))
    handler = NetHandler(VisionNet_1F, ((1 / 30) / 13) * 1000, img_size, 5)
    states = handler.init(input=imgs[0,0,:,:].unsqueeze(0))
    out = handler(imgs, states)
    print(out)