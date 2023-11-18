import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch as torch

def generate_ts():

    x = np.zeros((100, 1024))

    for i in range(100):
        x[i] = np.random.normal(loc=0.0, scale=1.0, size=1024)

    np.save('base_ts_100.npy', x)
    s = np.zeros((100 * 1000, 1025))
    j = 0
    for i in range(len(s)):
        h = np.random.normal(loc=0.0, scale=1.0, size=1000)
        for k in range(1000):
            t = x[i]*h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size = 1024)
            t += noise
            l = i
            s[j] = np.concatenate((t, np.array([l])), axis=0)
            j += 1

    np.save('train_ts_100.npy', s)

#generate_ts()

def generate_mix_ts(x, nametag):

    base_mix_ts = []
    base_labels = []
    mix_ts = []
    labels = []
    test_ts = []
    test_labels = []

    for i in range(len(x)):
        for j in range(len(x)):
            if j > i:
                z = np.random.choice([-1, 1], size=2)
                k1 = np.random.rand()
                k2 = (1 - k1)
                base_mix_ts.append(x[i] * k1 * z[0] + x[j] * k2 * z[1])
                base_labels.append([i, j])

    y = dict(mix_ts = base_mix_ts, labels = base_labels)
    scipy.io.savemat('base_mix_ts_%s.mat' % nametag, y)

    for i in range(len(base_mix_ts)):
        h = np.random.normal(loc=0.0, scale=1.0, size=1000)
        for j in range(1000):
            t = base_mix_ts[i] * h[j]
            noise = np.random.normal(loc=0.0, scale=0.04, size=1024)
            t += noise
            if j < 900:
                mix_ts.append(t)
                labels.append(base_labels[i])
            else:
                test_ts.append(t)
                test_labels.append(base_labels[i])

    s = dict(mix_ts=mix_ts, labels=labels)
    t = dict(mix_ts=test_ts, labels=test_labels)
    scipy.io.savemat('mix_ts_10_%s.mat' % nametag, s)
    scipy.io.savemat('test_ts_10_%s.mat' % nametag, t)

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

#stick_pattern()

#generate_mix_ts()
#x = scipy.io.loadmat('mix_ts.mat')
#print(len(x['mix_ts']))
#y = scipy.io.loadmat('test_ts.mat')
#print(len(y['mix_ts']))
"""
x= np.load('sticks_lib.npy',allow_pickle=True)
y = np.load('Q_300.npy',allow_pickle=True)
y = np.reshape(y,(-1,1,1,1))
for num in range(len(x)):
    basis = x[num]
    for (i, peak) in enumerate(basis):
        print(num,' ',i,' ', peak[0],' ', peak[1])
print(y)
"""
"""
array = np.arange(1, 21, 1, dtype = float)
x = np.zeros(20, dtype=float)
mean = [0,0,0,0,0,6,0,8,0,0,0,0,0,14,0,0,0,0,0,0]
deviation = [0,0,0,0,0,0.6,0,0.4,0,0,0,0,0,0.6,0,0,0,0,0,0]
peak = [0,0,0,0,0,-5,0,6.6,0,0,0,0,0,4.4,0,0,0,0,0,0]
ypoint = np.exp(-((array - 6)**2)/(2*0.6)) * -5
zpoint = np.exp(-((array - 8)**2)/(2*0.4)) * 6.6
xpoint = np.exp(-((array - 14)**2)/(2*0.6)) * 4.4
mpoint = np.exp(-((array - mean)**2)/(np.array(deviation) * 2 + 1e-9)) * peak
nopoint = zpoint + ypoint + xpoint
plt.subplot(1, 2, 1)
plt.plot(array, ypoint, array, zpoint,array, xpoint)
plt.subplot(1,2,2)
plt.plot(array, mpoint , array, nopoint)
plt.show()
"""
x = torch.tensor([[[[1],[2],[3]],[[1],[2],[3]]],[[[4],[5],[6]],[[4],[5],[6]]]])
print(x.size())
y = torch.tensor([[1,2],[5,6]])
print(y.size())
h = x.view(-1,3,1).permute(1,2,0) * y.view((-1))
h = h.permute(2,0,1).view(2,2,3,1)
print(h.size())
print(h)
#mat = scipy.io.loadmat('psl_NCEP2_C12_1979_2014_73x144_0.8_50_0.8.mat')
#print(len(mat['AllTs']))
#print(len(mat['AllTs'][0]))
#print(mat['AllTs'][0])
