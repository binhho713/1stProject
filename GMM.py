import torch
import numpy as np

def GMM(shifting_ratio, standard_deviation, amplitude_variance):

    #Load base peak location - amplitude pair
    base_ts =                        #amplitude
    base_mean =                      #peak location
    base_ts = torch.tensor(base_ts.reshape(-1, 1, 1024)).float()

    #Gaussian parameters
    x = np.arange(0, 1024, 1.0, dtype = float)
    x = torch.tensor(x.reshape(-1, 1, 1024)).float()
    variance = standard_deviation**2
    mean = base_mean * shifting_ratio
    peak = amplitude_variance * base_ts

    #Compute Gaussian
    output = torch.exp(-((x-mean)**2)/(2.0 * variance) - 0.5 * variance) * peak

    return output
