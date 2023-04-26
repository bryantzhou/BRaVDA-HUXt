###########################################################################################
# This part compares the RMSE from the variational method and the iEnKS method.
# It can also be used to compare the RMSE from any method.
###########################################################################################

# Import necessary packages
import os
import re
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

# Get the dierctory of data
currentDir = os.getcwd()
analysisDir = currentDir+'/Analysis/'

# Read data
rmseIenksDir = analysisDir+'rmse_ienks_20.txt'
rmseVarDir = analysisDir+'rmse_var.txt'

rmseIenks = open(rmseIenksDir,'r')
rmseVar = open(rmseVarDir,'r')

# Initialize the data array
staPostIenks = []
stbPostIenks = []

staPostVar = []
stbPostVar = []

# Extract the relevant data from the output file from the BRAVDA-HUXt
for i, line in enumerate(rmseIenks):
    if line[:23] == 'RMSE STEREO A Posterior':
        num = re.search(r"\d+\.\d+", line)
        number =  float(num.group())
        staPostIenks.append(number)
    elif line[:23] == 'RMSE STEREO B Posterior':
        num = re.search(r"\d+\.\d+", line)
        number =  float(num.group())
        stbPostIenks.append(number)

for i, line in enumerate(rmseVar):
    if line[:23] == 'RMSE STEREO A Posterior':
        num = re.search(r"\d+\.\d+", line)
        number =  float(num.group())
        staPostVar.append(number)
    elif line[:23] == 'RMSE STEREO B Posterior':
        num = re.search(r"\d+\.\d+", line)
        number =  float(num.group())
        stbPostVar.append(number)

lenMin = min(len(staPostIenks),len(staPostVar),len(stbPostVar),len(stbPostIenks))

figure(100)
plot(np.arange(1,lenMin+1),staPostIenks[:lenMin],marker='o',label='iEnKS')
plot(np.arange(1,lenMin+1),staPostVar[:lenMin],marker='o',label='variational')
xlabel('Window',fontsize=18)
ylabel('RMSE (km/s)',fontsize=18)
title(r'RMSE at STEREO A',fontsize=18)
xlim(0,lenMin+1)
legend(fontsize=8)
tight_layout()
grid()


figure(200)
plot(np.arange(1,lenMin+1),stbPostIenks[:lenMin],marker='o',label='iEnKS')
plot(np.arange(1,lenMin+1),stbPostVar[:lenMin],marker='o',label='variational')
xlabel('Window',fontsize=18)
ylabel('RMSE (km/s)',fontsize=18)
title(r'RMSE at STEREO B',fontsize=18)
xlim(0,lenMin+1)
legend(fontsize=8)
tight_layout()
grid()
# show()

rmseIenks.close()
rmseVar.close()

###########################################################################################
# This part plots the covariance 
###########################################################################################
close('all')
covDir = analysisDir+'cov_2.csv'
covariance = open(covDir,'r')

covarianceLines = covariance.readlines()

data = np.genfromtxt(covDir, delimiter=',')

plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

covariance.close()