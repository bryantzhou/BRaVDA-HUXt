import os
import re
import numpy as np
from matplotlib.pyplot import *

currentDir = os.getcwd()
analysisDir = currentDir+'/Analysis/'

rmseDir = analysisDir+'rmse_var.txt'

rmse = open(rmseDir,'r')

staPost = []
staPri = []

stbPost = []
stbPri = []

stbPostIenks20 = []
stbPostIenks30 = []

staPostVar = []
stbPostVar = []

for i, line in enumerate(rmse):
    if line[:23] == 'RMSE STEREO A Posterior':
        num = re.search(r"\d+\.\d+", line)
        number =  float(num.group())
        staPost.append(number)
    elif line[:22] == 'RMSE STEREO A Forecast':
        num = re.search(r"\d+\.\d+", line)
        number =  float(num.group())
        staPri.append(number)
    elif line[:23] == 'RMSE STEREO B Posterior':
        num = re.search(r"\d+\.\d+", line)
        number =  float(num.group())
        stbPost.append(number)
    elif line[:22] == 'RMSE STEREO B Forecast':
        num = re.search(r"\d+\.\d+", line)
        number =  float(num.group())
        stbPri.append(number)

staPost = np.array(staPost)
stbPost = np.array(stbPost)

staPost=np.append(staPost,[111,111,123,107,126,94,99,161,149,97,146,109,115,107,93,99,111,110])
stbPost=np.append(stbPost,[115,109,93,115,99,99,120,117,122,146,114,129,105,130,76,78,90,140])
# lenMin = min(len(staPostIenks20),len(staPostVar),len(stbPostVar),len(stbPostIenks20),len(stbPostIenks30),len(staPostIenks30))
# lenMin = len(stbPost)

print('Posterior mean = ',np.mean(staPost))
print('Prior mean = ',np.mean(stbPost))

