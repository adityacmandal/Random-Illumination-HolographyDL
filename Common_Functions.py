 import numpy as np

def unwrap_phase(x):
    y = x % (2 * np.pi)
    return torch.where(y > np.pi, 2*np.pi - y, y)

def centered_fft2(x):
    return np.fft.fftshift(np.fft.fft2(x))

def centered_ifft2(x):
    return np.fft.ifft2(np.fft.fftshift(x))

def phase_unwrapping(in_):
    f = np.zeros((600,600))
    for ii in range(600):
        for jj in range(600):
            x = ii - 600/2
            y = jj - 600/2
            f[ii,jj] = x**2 + y**2
    a = centered_ifft2(centered_fft2(np.cos(in_)*centered_ifft2(centered_fft2(np.sin(in_))*f))/(f+0.000001))
    b = centered_ifft2(centered_fft2(np.sin(in_)*centered_ifft2(centered_fft2(np.cos(in_))*f))/(f+0.000001))
    out = np.real(a - b)
    return out

def propagator(Nx, Ny, z, wavelength, deltaX, deltaY):
    k = 1/wavelength
    x = np.expand_dims(np.arange(np.ceil(-Nx/2), np.ceil(Nx/2), 1) * (1/(Nx*deltaX)), axis=0)
    y = np.expand_dims(np.arange(np.ceil(-Ny/2), np.ceil(Ny/2), 1) * (1/(Ny*deltaY)), axis=1)
    
    y_new = np.repeat(y, Nx, axis=1)
    x_new = np.repeat(x, Ny, axis=0)
    
    kp = np.sqrt(y_new**2 + x_new**2)
    term = k**2 - kp**2
    term = np.maximum(term, 0)
    phase = np.exp(1j * 2 * np.pi * z * np.sqrt(term))
    return phase


