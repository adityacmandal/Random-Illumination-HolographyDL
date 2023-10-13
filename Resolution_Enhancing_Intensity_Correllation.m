clc
clear all 
close all  
tic

%%%This MATLAB code is for reconstruction of hologram in intensity correlations rather than conventional intensity measurements. A hologram obained through second order correlations of intensity fluctuations provides a better quality reconstruction.

%MATLAB code for implementing intensity correlations for improving image quality of hologram

for l=1:150  %number of random patterns recorded
a0 = imread(sprintf('s%d.bmp',l)); %%read object speckle 
a1=im2double(a0);
IP=a1(1:1000,1:1000);  %%considering using 1000*1000 pixels of the random pattern 
cell{l}=IP;   %Store them in a cell
end
I_av = mean(cat(3, cell{:}), 3);  %Evaluate mean of all the random patterns; I_av  is the mean.

sum =0;
%reconstruction
for l=1:150
w=((cell{l}-I_av ).*(cell{l}-I_av ));  %Evaluate intensity correlation <ΔI(u)ΔI(u)>.
sum = sum+w;   %Sum for all the random patterns;

end
imagesc(abs(sum));  colormap('hot') %the 2D image of hologram obtained after intensity correlation
%sum is the final hologram obtained after implementation of correlation using 150 random patterns