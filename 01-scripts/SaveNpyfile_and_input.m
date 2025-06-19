% import numpy as np
% from scipy.io import savemat
% 
% # Load .npy files
% datall = np.load('../data/TexasData/datall_Texas1.npy')  # or concatenate all 6 parts if needed
% polall = np.load('../data/TexasData/polall_Texas.npy')
% 
% # Save as .mat
% savemat('texas_data.mat', {'datall': datall, 'polall': polall})

from scipy.io import loadmat
import numpy as np

# Load the .mat file
mat_data = loadmat('texas_data.mat')

# Access and save each variable as .npy
np.save('datall_Texas1.npy', mat_data['datall'])
np.save('polall_Texas.npy', mat_data['polall'])
