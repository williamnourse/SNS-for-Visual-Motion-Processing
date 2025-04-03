import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import argparse

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
INPUTS
"""
# General Params
seed = 1

# Signal Params
noise_std = 0.01
wavelength = 50
angle = torch.tensor(0)   # radians, CW
func = torch.sin    # sine wave
func = lambda x: torch.sign(torch.sin(x))   # square wave
amplitude = 0.5
vel = 100    # pixels/second

# Output Params
num_frames = 40
t_stop = 1


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DERIVED PARAMS
"""
generator = torch.Generator().manual_seed(1)

if camera == 'DAVIS':
    (height, width) = (260, 346)    # Resolution of DAVIS346 camera
elif camera == 'DVX':
    (height, width) = (480, 640)  # Resolution of DVXplorer S Duo camera
else:
    (height, width) = (7, 7)    # Minimum viable image

rows = torch.arange(0,height)
cols = torch.arange(0,width)

grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing='ij')

noise_mean = 0.0

dt = t_stop/num_frames    # s


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OUTPUT
"""
phase = -1*dt*vel/wavelength # units of 2pi radians


norm_img = mpl.colors.Normalize(vmin=0, vmax=1)
grating = amplitude*func(2*torch.pi*(grid_cols*torch.cos(angle) + grid_rows*torch.sin(angle))/wavelength + phase*(-2*torch.pi))/2 + 0.5
noisy = torch.clamp(grating + torch.randn(grating.size(), generator=generator)*noise_std + noise_mean,0,1)

def update(frame):
    global log_state_last
    x = frame*dt*vel
    phase = x/wavelength
    grating = amplitude*func(2*torch.pi*(grid_cols*torch.cos(angle) + grid_rows*torch.sin(angle))/wavelength + phase*(-2*torch.pi))/2 + 0.5
    noisy = torch.clamp(grating + torch.randn(grating.size(), generator=generator)*noise_std + noise_mean,0,1)
    img.set_data(noisy)
    ax_img.set(title=str(frame))

    
    return (img, evt)
    # plt.colorbar()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset of visual gratings")
    parser.add_argument('config', type=str, help='Path to the configuration file')