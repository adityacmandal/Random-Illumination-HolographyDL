# Random-Illumination-HolographyDL
Deep learning for holographic imaging with random illuminations. Uses autoencoders for twin image removal and blind single-shot reconstruction without needing ground truth.

The repository displays the associated code for the research paper:  [Manisha, Aditya Chandra Mandal, Mohit Rathor, Zeev Zalevsky, and Rakesh Kumar Singh. "Randomness assisted in-line holography with deep learning." Scientific Reports 13, no. 1 (2023): 10986](https://www.nature.com/articles/s41598-023-37810-w)

If you have any question, please contact the author: krakeshsingh.phy@iitbhu.ac.in 

Lab site : [Laboratory of Information Photonics & Optical Metrology](https://www.informationphotonics.com/)

## Abstract : 
We propose and demonstrate a holographic imaging scheme exploiting random illuminations for recording hologram and then applying numerical reconstruction and twin image removal. We use an in-line holographic geometry to record the hologram in terms of the second-order correlation and apply the numerical approach to reconstruct the recorded hologram. This strategy helps to reconstruct high-quality quantitative images in comparison to the conventional holography where the hologram is recorded in the intensity rather than the second-order intensity correlation. The twin image issue of the in-line holographic scheme is resolved by an unsupervised deep learning based method using an auto-encoder scheme. Proposed learning technique leverages the main characteristic of autoencoders to perform blind single-shot hologram reconstruction, and this does not require a dataset of samples with available ground truth for training and can reconstruct the hologram solely from the captured sample. Experimental results are presented for two objects, and a comparison of the reconstruction quality is given between the conventional inline holography and the one obtained with the proposed technique.

## File Lists:
- Resolution_Enhancing_Intensity_Correllation.m ( MATLAB code for implementing intensity correlations for improving image quality of hologram)
- Main.py

## Requirement:
- GPU memory > 8 GB
- Python version: 3.10.12
Installation:
PyTorch (version 2.0.1 with CUDA 11.8)

For Anaconda: `conda install pytorch==2.0.1 torchvision==0.15.2+cu118 -c pytorch`

For pip: `pip install torch===2.0.1 torchvision===0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html`

OpenCV for Python: `pip install opencv-contrib-python`

Torchsummary: `pip install torchsummary`

## Here are the optical parameters provided in `Main.py`:
Spherical light function 
- Hologram size Nx, Ny: Nx = 600, Ny = 600
- Object-sensor distance z: z = 200000 (um)
- wavelength: wavelength = 0.6328 (um)
-  pixel size, deltaX, deltaY: deltaX = 5.3 , deltaY = 5.3
- To configure your parameters, please access the main.py file and adjust them

## Training:

Model was implemented using the PyTorch framework and runs on a GPU workstation with an Nvidia tesla P100 graphics card. The Adam optimization algorithm is utilized with a fixed learning rate of 0.005. Use the training process for 4000 iterations until we get the final reconstruction. The entire training process takes approximately 731 seconds to complete.
  
## Experimental Setup
<img width="626" alt="Screenshot 2023-10-13 at 4 43 59 PM" src="https://github.com/adityacmandal/Random-Illumination-HolographyDL/assets/95050827/21ebb0f2-2a54-46ac-810a-3657c6ead6d3">

SF: Spatial filter, L1 and L2:Lens with focal lengths 200 and 100 mm respectively, BS: Beam splitter, SLM: Spatial light modulator, RGG: Rotating ground glass, A1 and A2:Apertures, CMOS Camera: Complementary metal oxide semiconductor camera [[paper].](https://www.nature.com/articles/s41598-023-37810-w).

<img width="1040" alt="Screenshot 2023-10-13 at 4 44 44 PM" src="https://github.com/adityacmandal/Random-Illumination-HolographyDL/assets/95050827/abf33619-e744-448a-81af-e474635d9407">

A flow chart for the proposed scheme [[paper].](https://www.nature.com/articles/s41598-023-37810-w).

<img width="1073" alt="Screenshot 2023-10-13 at 4 44 22 PM" src="https://github.com/adityacmandal/Random-Illumination-HolographyDL/assets/95050827/b4a3491a-d936-4cac-8190-c20441679fb1">

An auto-encoder network utilizing deep convolutional layers in an hourglass architecture [[paper].](https://www.nature.com/articles/s41598-023-37810-w).


