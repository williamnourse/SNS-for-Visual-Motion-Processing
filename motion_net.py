import torch
import torch.nn as nn

def process_events(event_frame: torch.Tensor, merge=True):
    """
    Converts an event frame into a split frame, one for positive and one for negative edges
    :param event_frame: input frame, assuming (batch, height, width) format
    :param merge: If True, combine into one tensor. If False, output positive and negative channels as separate images
    :return: split_frame, (batch, 2, height, width), or pos and neg
    """
    shape = event_frame.shape
    if merge:
        split_frame = torch.zeros([shape[0],2,shape[1], shape[2]])
        split_frame[:,0,:,:] = event_frame
        split_frame[:,1,:,:] = -event_frame
        nn.functional.relu(split_frame, inplace=True)
        return split_frame
    else:
        pos = nn.functional.relu(event_frame)
        neg = nn.functional.relu(-event_frame)
        return pos, neg

def process_images(img_frame: torch.Tensor, merge=True):
    """
    Converts an event frame into a split frame, one for positive and one for negative edges
    :param img_frame: input frame, assuming (batch, height, width) format
    :param merge: If True, combine into one tensor. If False, output original and inverted channels as separate images
    :return: split_frame, (batch, 2, height, width), or original and inverted
    """
    shape = img_frame.shape
    if merge:
        split_frame = torch.zeros([shape[0],2,shape[1], shape[2]])
        split_frame[:,0,:,:] = img_frame
        split_frame[:,1,:,:] = 1 - img_frame
        return split_frame
    else:
        inv_frame = 1 - img_frame
        return img_frame, inv_frame

class TraceLayer(nn.Module):
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

    def forward(self, x, state):
        return state - self.params['tau'].view(1,self.num_channels, 1, 1) + x

class LILayer(nn.Module):
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

    def forward(self, x, state):
        return state + self.params['tau'].view(1,self.num_channels, 1, 1) * (x - state)

class LI2Layer(nn.Module):
    def __init__(self, num_channels, params=None, device=None, dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            if device is None:
                device = 'cpu'
            self.params = nn.ParameterDict({
                'tau0': nn.Parameter(
                    torch.rand(num_channels, device=device, dtype=dtype, generator=generator).to(device)),
                'tau1': nn.Parameter(
                    torch.rand(num_channels, device=device, dtype=dtype, generator=generator).to(device))
            })
            self.num_channels = num_channels
            if params is not None:
                self.params.update(params)

    def forward(self, x, state_0, state_1):
        state_0_new = state_0 + self.params['tau0'].view(1,self.num_channels, 1, 1) * (x - state_0)
        state_1_new = state_1 + self.params['tau1'].view(1,self.num_channels, 1, 1) * (state_0 - state_1)
        return state_1_new - state_0_new, state_0_new, state_1_new

class ShuntedLILayer(nn.Module):
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

    def forward(self, x, shunt, state):
        return state + self.params['tau'].view(1,self.num_channels, 1, 1) * (x - state * (1 + shunt))

class MotionNet(nn.Module):
    def __init__(self, event_channels=2, img_channels=2, num_filters_event=6, num_filters_img=2, num_emd=8, kernel_size=5):
        super().__init__()
        # Event pathway
        self.conv_e = nn.Conv2d(event_channels, num_filters_event, kernel_size=kernel_size)
        self.trace = TraceLayer(num_filters_event)
        self.li_e = LILayer(num_filters_event)
        self.clip_e = nn.Hardtanh(min_val=0, max_val=1)

        # Frame pathway
        self.conv_i = nn.Conv2d(img_channels, num_filters_img, kernel_size=kernel_size)
        self.li_i = LILayer(num_filters_img)
        self.clip_i = nn.Hardtanh(min_val=0, max_val=1)

        # EMDs
        self.conv_emd = nn.Conv2d(num_filters_event, num_emd, kernel_size=3)
        self.conv_emd_img = nn.Conv2d(num_filters_img, num_emd, kernel_size=3)
        self.emd = ShuntedLILayer(num_emd)

    def forward(self, event_frame, img, state_trace, state_li_e, state_li_i, state_emd):
        # Events
        conv_e = self.conv_e(event_frame)
        self.trace(conv_e, state_trace)
        self.li_e(state_trace, state_li_e)
        out_e = self.clip_e(state_li_e)

        # Image frames
        conv_i = self.conv_i(img)
        self.li_i(conv_i, state_li_i)
        out_i = self.clip_i(state_li_i)

        # EMD
        direct = self.conv_emd(out_e)
        shunt = self.conv_emd_img(out_i)
        self.emd(direct, shunt, state_emd)
        # print(state_emd[0,0,0,0])

        return state_trace, state_li_e, state_li_i, state_emd


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import time
    from tqdm import tqdm

    with torch.no_grad():
        input_low = torch.tensor(0.0)
        input_high = torch.tensor(1.0)
        nested = LI2Layer(1)
        num_steps = 100
        data0 = torch.zeros(num_steps)
        data1 = torch.zeros(num_steps)
        dataOut = torch.zeros(num_steps)
        state_0 = torch.tensor(0.0)
        state_1 = torch.tensor(0.0)
        state_out = torch.tensor(0.0)
        for i in range(num_steps):
            if 10 < i < 50:
                state_out, state_0, state_1 = nested(input_high, state_0, state_1)
                # if state_out > 0:
                #     pass
            else:
                state_out, state_0, state_1 = nested(input_low, state_0, state_1)
            data0[i] = state_0
            data1[i] = state_1
            dataOut[i] = state_out
        plt.figure()
        plt.plot(data0)
        plt.plot(data1)
        plt.plot(dataOut)
        plt.show()
    #
    #     height, width = 260, 346
    #     # height, width = 480, 640
    #
    #     example_event_frame = torch.randint(0,3,[1,height,width]) -1.0
    #     print(example_event_frame)
    #     split_example_frame = process_events(example_event_frame)
    #     print(split_example_frame)
    #
    #     example_img = torch.rand([1,height,width])
    #     # print(example_img)
    #     split_img = process_images(example_img)
    #     # print(split_img)
    #     # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    #     # plt.figure()
    #     # plt.subplot(1,2,1)
    #     # plt.imshow(split_img[0,0,:,:], cmap='Greys', norm=norm, interpolation='none')
    #     # plt.subplot(1,2,2)
    #     # plt.imshow(split_img[0,1,:,:], cmap='Greys', norm=norm, interpolation='none')
    #     # plt.show()
    #     event_channels = 2
    #     img_channels = 2
    #     num_filters_event = 6
    #     num_filters_img = 2
    #     num_emd = 8
    #
    #     state_trace = torch.rand([1,num_filters_event,height-4,width-4])
    #     state_li_e = torch.rand([1,num_filters_event,height-4,width-4])
    #     state_li_i = torch.rand([1,num_filters_img,height-4,width-4])
    #     state_emd = torch.rand([1,num_emd,height-6,width-6])
    #
    #     net = MotionNet(event_channels=event_channels, img_channels=img_channels, num_filters_event=num_filters_event, num_filters_img=num_filters_img, num_emd=num_emd, kernel_size=5)
    #     num_samples = 1000
    #     start = time.time()
    #     for i in tqdm(range(num_samples)):
    #         state_trace, state_li_e, state_li_i, state_emd = net(split_example_frame, split_img, state_trace, state_li_e, state_li_i, state_emd)
    #     end = time.time()
    #     print('Finished %i in %i secs'%(num_samples,end-start))
    #     print('Avg step time: %.6f'%((end-start)/num_samples))
