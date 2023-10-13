# Random-Illumination-HolographyDL
Deep learning for holographic imaging with random illuminations. Uses autoencoders for twin image removal and blind single-shot reconstruction without needing ground truth.

Paper : Manisha, Aditya Chandra Mandal, Mohit Rathor, Zeev Zalevsky, and Rakesh Kumar Singh. "Randomness assisted in-line holography with deep learning." Scientific Reports 13, no. 1 (2023): 10986. ( https://www.nature.com/articles/s41598-023-37810-w )

Abstract : We propose and demonstrate a holographic imaging scheme exploiting random illuminations for recording hologram and then applying numerical reconstruction and twin image removal. We use an in-line holographic geometry to record the hologram in terms of the second-order correlation and apply the numerical approach to reconstruct the recorded hologram. This strategy helps to reconstruct high-quality quantitative images in comparison to the conventional holography where the hologram is recorded in the intensity rather than the second-order intensity correlation. The twin image issue of the in-line holographic scheme is resolved by an unsupervised deep learning based method using an auto-encoder scheme. Proposed learning technique leverages the main characteristic of autoencoders to perform blind single-shot hologram reconstruction, and this does not require a dataset of samples with available ground truth for training and can reconstruct the hologram solely from the captured sample. Experimental results are presented for two objects, and a comparison of the reconstruction quality is given between the conventional inline holography and the one obtained with the proposed technique.

<img width="943" alt="Screenshot 2023-10-13 at 4 43 59 PM" src="https://github.com/adityacmandal/Random-Illumination-HolographyDL/assets/95050827/b0080a7e-fb20-48ed-9ac9-af11636502bf">

<img width="1046" alt="Screenshot 2023-10-13 at 4 44 44 PM" src="https://github.com/adityacmandal/Random-Illumination-HolographyDL/assets/95050827/8bc2728e-6a25-469b-b6ef-86205d977f98">

<img width="819" alt="Screenshot 2023-10-13 at 4 45 02 PM" src="https://github.com/adityacmandal/Random-Illumination-HolographyDL/assets/95050827/0fa332d4-1294-4543-82ac-eb3d74026940">
