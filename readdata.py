import scipy.io
import numpy as np
import torch as torch
import scipy.stats.mstats as statm
import pandas as pd
import random
import logging
import time
import itertools

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with a file and console handler."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger

def generate_ts(size, length):

    x = np.zeros((size, length))

    for i in range(size):
        x[i] = np.random.normal(loc=0.0, scale=1.0, size=length)

    np.save('base_ts_%s.npy' % size, x)
    s = np.zeros((size * 1000, length))
    j = 0
    for i in range(len(s)):
        h = np.random.normal(loc=0.0, scale=1.0, size=length)
        for k in range(length):
            t = x[i]*h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size=length)
            t += noise
            l = i
            s[j] = np.concatenate((t, np.array([l])), axis=0)
            j += 1

    np.save('train_ts_100.npy', s)

def generate_train_ts(data, nametag, labels, s_or_r):
    size = data.shape[0]
    length = data.shape[1]
    s = np.zeros((size * 1000, length + 1))
    j = 0
    l = []
    x = statm.zscore(data, axis=1)

    for i in range(size):
        h = np.random.normal(loc=0.0, scale=1.0, size=1000)
        for k in range(1000):
            t = x[i]*h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size=length)
            t += noise
            l = i
            s[j] = np.concatenate((t, np.array([l])), axis=0)
            #s[j] = t
            #l.append(labels[i])
            j += 1

    if s_or_r:
        np.save(nametag, s)
    else:
        return s, l

def generate_train_ts_with_labels(input, labels, nametag, sample, l_opt, s_or_r):

    x = statm.zscore(input, axis=0)
    size = x.shape[0]
    length = x.shape[1]
    if l_opt:
        s = np.zeros((size * sample, length + 3))
    else:
        s = np.zeros((size * sample, length))
    j = 0
    for i in range(size):
        h = np.random.normal(loc=0.0, scale=1.0, size=sample)
        for k in range(sample):
            t = x[i]*h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size=length)
            t += noise
            if l_opt:
                l = labels[i]
                s[j] = np.concatenate((t, np.array([l])[0]), axis=0)
            else:
                s[j] = t
            j += 1

    s = statm.zscore(s, axis=1)
 
    if s_or_r:
        np.save(nametag, s)
    else:
        return s

def generate_mixture(x, k):
    mixture = []
    labels = []
    indice = np.arange(0, len(x), 1)
    arr = np.zeros((k, x.shape[1]))

    for combo in itertools.combinations(indice, k):
        for i in range(k):
            arr[i] = x[int(combo[i])]
        z = np.random.choice([-1, 1], size=k)
        coef = np.random.rand(k)
        coef *= z
        coef /= sum(coef)
        arr *= coef[:, np.newaxis]
        mixture.append(sum(arr))
        labels.append(combo)

    return mixture, labels

def generate_mix_ts(x, nametag, sample, train_ratio, l_opt, s_or_r):
    base_mix_ts = []
    base_labels = []
    mix_ts = []
    labels = []
    test_ts = []
    test_labels = []
    
    #generate base mixture
    for i in range(len(x)):
        for j in range(len(x)):
            if j > i:
                z = np.random.choice([-1, 1], size=2)
                k1 = np.random.rand()
                k2 = (1 - k1)
                base_mix_ts.append(x[i] * k1 * z[0] + x[j] * k2 * z[1])
                base_labels.append([i, j])

    y = dict(mix_ts=base_mix_ts, labels=base_labels)
    scipy.io.savemat('base_mix_ts_%s.mat' % nametag, y)

    #generate mixture samples
    for i in range(len(base_mix_ts)):
        h = np.random.normal(loc=0.0, scale=1.0, size=sample)
        for j in range(sample):
            t = base_mix_ts[i] * h[j]
            noise = np.random.normal(loc=0.0, scale=0.04, size=(x.shape[1]))
            t += noise
            if j < sample * train_ratio:
                mix_ts.append(t)
                if l_opt:
                    labels.append(base_labels[i])
            else:
                test_ts.append(t)
                if l_opt:
                    test_labels.append(base_labels[i])
	
    if l_opt:
    	s = dict(mix_ts=mix_ts, labels=labels)
    	t = dict(mix_ts=test_ts, labels=test_labels)
    if s_or_r:
        if l_opt:
            scipy.io.savemat(nametag, s)
            scipy.io.savemat('test' + nametag, t)
        else:
            np.save(nametag, s)
            np.save('test' + nametag, t)
    else:
        if l_opt:
            return s, t
        else:
            return mix_ts, labels

def gaussian(peak, mu, standard_deviation):
    # peak 200, 1
    # mu 200, 1
    # std 1
    variance = standard_deviation**2
    x = np.arange(0, 1024, 1)
    output = np.exp(-((x - mu)**2)/(2.0 * variance)) * peak #200, 1024
    output = np.sum(output, axis=0)

    return output

def stick_pattern():

    peak = np.zeros(shape= (10,200,1), dtype='float')
    mu = np.zeros(shape= (10,200,1), dtype='float')
    standard_deviation = np.random.uniform(size=10) + 1e-6
    np.save('standard_deviation.npy', standard_deviation)
    set = np.load('sticks_lib.npy',allow_pickle=True)
    for num in range(10):
        for (i, k) in enumerate(set[num]):
            mu[num][i] = k[0]
            peak[num][i] = k[1]

    stickpattern = dict(mu = mu, peak = peak)
    scipy.io.savemat('stick_patterns.mat', stickpattern)
    ts = []

    for i in range(len(peak)):
        ts.append(gaussian(peak[i], mu[i], standard_deviation[i]))

    np.save('ts_base_GMM', ts)

    s = np.zeros((10 * 1000, 1025))
    j = 0
    for i in range(len(ts)):
        h = np.random.normal(loc=0.0, scale=1.0, size=1000)
        for k in range(1000):
            t = ts[i] * h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size=1024)
            t += noise
            l = i
            s[j] = np.concatenate((t, np.array([l])), axis=0)
            j += 1

    np.save('train_ts_10_GMM.npy', s)

    generate_mix_ts(ts, 'GMM')

def Brute_force(dataPath, k, threshold, run_time):
    duration = run_time * 60 * 60
    start_time = time.time()
    # Load data with memory mapping
    data = np.load(dataPath, mmap_mode='r')

    # Create a range of indices for combinations
    cb = range(len(data))
    mlp = []
    strength = []
    arr = np.zeros((k, data.shape[1]))
    st = True
    print('Start mining multipoles in ' + dataPath)
    # Iterate over all combinations of k indices
    for i in itertools.combinations(cb, int(k)):
        for j in range(k):
            arr[j] = data[i[j]]

        arr = statm.zscore(arr, axis=1)
        x = np.transpose(arr)
        x = np.corrcoef(x, rowvar=False)
        x = np.nan_to_num(x)

        eigenvalues, eigenvectors = np.linalg.eig(x)
        min_variance_index = np.argmin(eigenvalues)
        min_variance_eigenvector = eigenvectors[:, min_variance_index]

        s = 0
        for j in range(k):
            s += arr[j] * min_variance_eigenvector[j]

        if np.var(s) <= threshold:
            mlp.append(i)
            strength.append(np.var(s))

        # Check if the specified run time has elapsed
        current_time = time.time()
        if (current_time - start_time) >= duration:
            st = False
            print('Breaking loop due to time constraint.')
            break
    # Save the results to a .mat file
    print('run for %d' % (time.time() - start_time))
    output = dict(mlp=mlp, strength= strength, finished=st, runTime=(time.time() - start_time))
    scipy.io.savemat('brute_force_%d.mat' % len(data), output)
    print('Finished')

base_psl = np.load("base_ts_171_psl.npy", allow_pickle=True)
base_psl = statm.zscore(base_psl, axis=1)
base_asv = np.load("base_ts_200_ASV.npy", allow_pickle=True)
base_asv = statm.zscore(base_asv, axis=1)

generate_train_ts(base_psl, nametag = 'train_cGAN_psl.npy', labels= None, s_or_r = True)
generate_train_ts(base_asv, nametag = 'train_cGAN_ASV.npy', labels= None, s_or_r = True)
