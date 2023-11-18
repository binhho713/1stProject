import torch
import numpy as np

def GMM(peak, mu, c):

    #peak bs, ts, 200, 1
    #mu bs, ts, 200, 1
    #c bs, ts, 1
    #Gaussian parameters
    x = np.arange(0, 1024, 1.0, dtype = float)
    x = torch.tensor(x.reshape(-1, 1, 1024)).float().cuda()
    variance = (c**2) + 1e-9 #bs, ts, 1
    output = torch.zeros(64, 10, 1024).cuda() #bs, ts, 1024

    #Compute Gaussian
    for i in range(output.shape[0]):
        for k in range(output.shape[1]):
            g = torch.exp(-((x-mu[i][k])**2)/(2.0 * variance[i][k])) * peak[i][k] #1, 100, 1024
            g = torch.sum(g, dim=1) # 1, 1024
            output[i][k] = g

    return output
