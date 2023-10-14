import torch
import torch.nn as nn
import numpy as np

# Propagation-based reconstruction loss
class PropagationLoss(nn.Module):
    def __init__(self):
        super(PropagationLoss, self).__init__()
        self.Nx = 600
        self.Ny = 600
        self.z = 200000
        self.wavelength = 0.6328
        self.deltaX = 5.3
        self.deltaY = 5.3
        self.prop = self._compute_propagator(self.Nx, self.Ny, self.z, self.wavelength, self.deltaX, self.deltaY).cuda()

    def _compute_propagator(self, Nx, Ny, z, wavelength, deltaX, deltaY):
        """Compute phase propagation."""
        k = 1 / wavelength
        x = np.expand_dims(np.arange(np.ceil(-Nx/2), np.ceil(Nx/2), 1) * (1 / (Nx * deltaX)), axis=0)
        y = np.expand_dims(np.arange(np.ceil(-Ny/2), np.ceil(Ny/2), 1) * (1 / (Ny * deltaY)), axis=1)
        y_new, x_new = np.repeat(y, Nx, axis=1), np.repeat(x, Ny, axis=0)
        term = k**2 - (y_new**2 + x_new**2)
        term = np.maximum(term, 0)
        phase = np.exp(1j * 2 * np.pi * z * np.sqrt(term))
        return torch.from_numpy(np.concatenate([np.real(phase)[np.newaxis, :, :, np.newaxis], np.imag(phase)[np.newaxis, :, :, np.newaxis]], axis=3))

    # Various utility methods for handling complex numbers, rolling, and FFT operations
    def complex_mult(self, x, y):
        real_part = x[:,:,:,0]*y[:,:,:,0]-x[:,:,:,1]*y[:,:,:,1]
        real_part = real_part.unsqueeze(3)
        imag_part = x[:,:,:,0]*y[:,:,:,1]+x[:,:,:,1]*y[:,:,:,0]
        imag_part = imag_part.unsqueeze(3)
        return torch.cat((real_part, imag_part), 3)

    def forward(self,x,y):
        x = x.squeeze(2)
        y = y.squeeze(2)
        x = x.permute([0,2,3,1])
        y = y.permute([0,2,3,1])

        print(x.shape)
        cEs = torch.fft.fftshift(torch.fft.fftn(x, dim=(1, 2), norm="ortho"), dim=(1, 2))
        cEsp = self.complex_mult(cEs, self.prop)

        S = torch.fft.ifftn(torch.fft.ifftshift(cEsp, dim=(1, 2)), dim=(1, 2), norm="ortho")
        Se = S[:, :, :, 0]


        mse = torch.mean(torch.abs(Se-y[:,:,:,0]))/2
        return mse

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


# Discrete Wavelet Transform
def discrete_wavelet_transform(x):
    """Perform a 2D discrete wavelet transform."""
    # Transformation logic
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    x_transformed = torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

    return x_transformed

# Inverse Discrete Wavelet Transform
def inverse_discrete_wavelet_transform(x):
    """Perform an inverse 2D discrete wavelet transform."""
    # Transformation logic
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    x_reconstructed = h

    return x_reconstructed

# Neural Network for Wave Propagation Tasks
class WavePropagationNet(nn.Module):
    def __init__(self):
        super(WavePropagationNet, self).__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
        )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 16, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
        )
        
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 2, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = x.float()
        x = self.conv_init(x)
        x = discrete_wavelet_transform(x)
        x = self.conv_1(x)
        x = discrete_wavelet_transform(x)
        x = self.conv_2(x)
        x = discrete_wavelet_transform(x)
        x = self.conv_nonlinear(x)
        
        x = self.deconv_1(x)
        x = inverse_wavelet_transform(x)
        x = self.deconv_2(x)
        x = inverse_wavelet_transform(x)
        x = self.deconv_3(x)
        x = inverse_wavelet_transform(x)
        x = self.deconv_4(x)
        
        return x

