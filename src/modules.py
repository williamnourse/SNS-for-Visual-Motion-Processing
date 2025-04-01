import torch
import torch.nn as nn
import torch.nn.functional as F


class LI(nn.Module):
    def __init__(self, num_channels, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'tau': nn.Parameter(torch.rand(num_channels, device=device, dtype=dtype, generator=generator).to(device))
        })
        self.num_channels = num_channels
        if params is not None:
            self.params.update(params)
        self.clamp = nn.Hardtanh(min_val=0, max_val=1)

    def forward(self, x, state):
        state_new = state + self.params['tau'].view(1,self.num_channels, 1, 1) * (x - state)
        return self.clamp(state_new), state_new


class LI2(nn.Module):
    def __init__(self, num_channels, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            if device is None:
                device = 'cpu'
            self.params = nn.ParameterDict({
                'tau0': nn.Parameter(
                    torch.rand(num_channels, device=device, dtype=dtype, generator=generator).to(device)),
                'tau1': nn.Parameter(
                    torch.rand(num_channels, device=device, dtype=dtype, generator=generator).to(device)),
                'bias': nn.Parameter(
                    torch.rand(num_channels, device=device, dtype=dtype, generator=generator).to(device)),
                'gate0': nn.Parameter(
                    torch.randn(num_channels, device=device, dtype=dtype, generator=generator).to(device)),
                'gate1': nn.Parameter(
                    torch.randn(num_channels, device=device, dtype=dtype, generator=generator).to(device)),
            })
            self.num_channels = num_channels
            if params is not None:
                self.params.update(params)
            self.clamp = nn.Hardtanh(min_val=0, max_val=1)

    def forward(self, x, state_0, state_1):
        state_0_new = state_0 + self.params['tau0'].view(1, self.num_channels, 1, 1) * (x - state_0)
        state_1_new = state_1 + self.params['tau1'].view(1, self.num_channels, 1, 1) * (state_0 - state_1)
        out = state_0_new * self.params['gate0'].view(1, self.num_channels, 1, 1) +\
              state_1_new * self.params['gate1'].view(1, self.num_channels, 1, 1) +\
              self.params['bias'].view(1, self.num_channels, 1, 1)
        # out = state_0_new * self.params['gate0'] + state_1_new * self.params['gate1'] + self.params['bias']
        return self.clamp(out), state_0_new, state_1_new


class MotionNet(nn.Module):
    def __init__(self, input_channels=1, num_filters_0=2, num_filters_1=4, num_emd=4, kernel_size=3):
        super().__init__()
        # Layer 0 (Lamina)
        self.conv_0 = nn.Conv2d(input_channels, num_filters_0, kernel_size=kernel_size)
        self.li2_0 = LI2(num_channels=num_filters_0)

        # Layer 1 (Medulla)
        self.conv_1 = nn.Conv2d(num_filters_0, num_filters_1, kernel_size=kernel_size)
        self.li_1 = LI(num_channels=num_filters_1)

        # EMDs (Lobula/Lopula Plate)
        self.conv_emd = nn.Conv2d(num_filters_1, num_emd, kernel_size=3)
        self.conv_shunt = nn.Conv2d(num_filters_1, num_emd, kernel_size=3)
        self.emd = LI(num_emd)

    def forward(self, x, state_0_0, state_0_1, state_1, state_emd):
        # Layer 0 (Lamina)
        x = self.conv_0(x)
        x, state_0_0, state_0_1 = self.li2_0(x, state_0_0, state_0_1)

        # Layer 1 (Medulla)
        x = self.conv_1(x)
        x, state_1 = self.li_1(x, state_1)

        # EMD (Lobula/Lobula Plate)
        direct = self.conv_emd(x)
        shunt = self.conv_shunt(x)
        x = direct + shunt*(-state_emd)
        x, state_emd = self.emd(x, state_emd)
        # print(state_emd[0,0,0,0])

        return x, state_0_0, state_0_1, state_1, state_emd


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import time
    from tqdm import tqdm

    test_network = False
    test_loss = True
    if test_network:
        with torch.no_grad():
            height, width = 32, 34
            num_filters_0 = 2
            num_filters_1 = 4
            num_emd = 4
            kernel_size=3
            num_steps = 10000

            example_img = torch.rand([1, height, width])

            net = MotionNet(num_filters_0=num_filters_0, num_filters_1=num_filters_1, num_emd=num_emd,
                            kernel_size=kernel_size)

            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            plt.figure()
            plt.imshow(example_img[0,:,:], cmap='Greys_r', norm=norm, interpolation='none')
            plt.colorbar()

            state_0_0 = torch.zeros(1, num_filters_0, height-kernel_size+1, width-kernel_size+1)
            state_0_1 = torch.zeros(1, num_filters_0, height-kernel_size+1, width-kernel_size+1)
            state_1 = torch.zeros(1, num_filters_1, height-kernel_size-1, width-kernel_size-1)
            state_emd = torch.zeros(1, num_emd, height-kernel_size-3, width-kernel_size-3)

            start = time.time()
            for i in tqdm(range(num_steps)):
                out, state_0_0, state_0_1, state_1, state_emd = net(example_img, state_0_0, state_0_1, state_1, state_emd)
            end = time.time()
            print('Avg step: %.4f secs'%((end-start)/num_steps))

            plt.show()

    if test_loss:
        b = 4
        targets = torch.tensor([[1.0,0.0],
                                [0.0,1.0],
                                [-1.0,0.0],
                                [0.0,-1.0]])
        data = torch.tensor([[0.0,0.0,0.0,1.0],
                             [1.0,0.0,0.0,1.0],
                             [0.0,0.0,1.0,0.0],
                             [0.0,1.0,0.0,0.0]])
        data = data.reshape(4,4,1,1)
        loss = angular_field_loss(data,targets)
        print(loss)
