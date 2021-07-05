import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import mixfit
#Fixing seed for easy comparison
np.random.seed(1)

def test(tau, mu1, sigma1, mu2, sigma2): 
    '''
    Test preparation routine
    
    Parameters
    ----------
        tau : float
            Relative number of objects in the first distribution
        mu1 : float
            Mean of the first distribution
        sigma1 : float
            Dispersion of the first distribution
        mu2 : np.array with shape (2,)
            Mean of the second distribution
        sigma2 : float
            Dispersion of the second distribution
            
    Returns
    -------
        np.array
            Test data
    '''
    k = 10000 
    n1 = stats.norm.rvs(loc=mu1, scale=sigma1, size=int(k * tau)) 
    n2 = stats.norm.rvs(loc=mu2, scale=sigma2, size=int(k * (1 - tau)))
    return np.concatenate((n1, n2))

if __name__ == "__main__":
    tau = 0.5
    mu1 = 0.1
    sigma1 = 0.4
    mu2 = 0.5
    sigma2 = 0.3 

    data = test(tau, mu1, sigma1, mu2, sigma2)

    '''Test 1'''

    with open("test1.txt", "w") as f:
        print('data_shape: ', data.shape, file=f)
        x1 = mixfit.max_likelihood(data, 0.3, 0.3, 0.5, 0.3, 0.4)
        print('max_likelihood: ', np.round(x1, 5), file=f)
        x2 = mixfit.em_double_gauss(data, 0.3, 0.3, 0.5, 0.3, 0.4)
        print('EM_1 method: ', np.round(x2, 5), file=f)

    '''Test 2'''

    tau = 0.4
    mu1 = np.array([0.5, 0.5])
    mu2 = np.array([-0.5, -0.5])
    sigma1 = 0.4
    sigma2 = 0.2
    n = 10000
    n_1 = int(n * tau)
    n_2 = n - n_1

    x_n1 = stats.multivariate_normal(mu1, sigma1**2).rvs(n_1)
    x_n2 = stats.multivariate_normal(mu2, sigma2**2).rvs(n_2)
    x = np.vstack((x_n1, x_n2))
    plt.hist2d(*x.T)
    plt.plot(*x.T, '.', color='blue')

    with open("test2.txt", "w") as f:
        print('EM_1 method_X: ', np.round(mixfit.em_double_gauss(x[:, 0], \
                             0.6, -0.5, 0.3, 0.5, 0.4), 5), file=f)
        print('EM_1 method_Y: ', np.round(mixfit.em_double_gauss(x[:, 1], \
                             0.6, -0.5, 0.3, 0.5, 0.4), 5), file=f)
    plt.savefig("test.png")
