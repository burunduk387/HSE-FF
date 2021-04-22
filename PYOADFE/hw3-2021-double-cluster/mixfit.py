import numpy as np
from scipy import stats, optimize

def density_function(args, x):
    '''
    Density functions for 2 normal distributions
    
    Parameters
    ----------
        args : object
            Contains all the neccesary data, as stated below:
        tau : float
            Relative numbers of stars in the first and second clusters
        mu1 : float
            Average position of the first distribution
        sigma1 : float
            The scatter of x in the first distribution
        mu2 : float
            Average position of the second distribution
        sigma2 : float
            The scatter of x in the second distribution
        rtol : float
            The relative tolerance parameter
            
    Returns
    -------
        res : np.array
            Sum of all density functions, multiplied by their weights
        density_1 : np.array
            Density functions values for the first distribution (normalized)
        density_2 : np.array
            Density functions values for the second distribution (normalized)
        density_3 : np.array
            Density functions values for the third distribution (normalized)
    '''
    tau, mu1, sigma1, mu2, sigma2 = args
    density_1 = stats.norm.pdf(x, loc=mu1, scale=np.abs(sigma1))
    density_2 = stats.norm.pdf(x, loc=mu2, scale=np.abs(sigma2))
    res = tau * density_1 + (1 - tau) * density_2
    return res, density_1, density_2

def optimization_function(args, x):
    '''
    Function used for making my life with optimize.minimize easier

    Parameters
    ----------
        args : object
            Contains all the neccesary data, as stated below:
        tau : float
            Relative numbers of stars in the first and second clusters
        muv : np.array with shape (2,)
            Mean proper motion of the cluster
        mu1 : np.array with shape (2,)
            Average position of the first cluster stars
        mu2 : np.array with shape (2,)
            Average position of the second cluster stars
        sigma02 : float
            The scatter of the proper motions of the stars of the field
        sigmax2 : float
            The scatter of the positions of cluster stars around the mean
        sigmav2 : float
            The scatter of proper motions of cluster stars relative to the mean
        rtol : float
            The relative tolerance parameter
    Returns
    -------
        value : float
            Value of the function

    '''
    
    #Поскольку оптимизируем при фиксированных x
    #Фактически имеем дело с func(tau, mu1, ..., const_arg=x)
    #То есть как бы вместо "икса" в традиционном понимании f(x) 
    #У меня всякие tau, mu1, ..., надеюсь ниже по коду понятнее
    res = density_function(args, x)[0]
    #Если перед функцией, которую надо максимизировать, поставить минус
    #То её надо минимизировать, а это как раз мы и умеем
    return -np.sum(np.log(res))



def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    '''
    Maximum likelihood method
    
    Parameters
    ----------
        x : np.array
            Contains values of x
        tau : float
            Relative number of objects in the first distribution
        mu1 : float
            Average position of the first distribution
        sigma1 : float
            The scatter of x in the first distribution
        mu2 : float
            Average position of the second distribution
        sigma2 : float
            The scatter of x in the second distribution
        rtol : float
            The relative tolerance parameter
            
    Returns
    -------
        tau, mu1, sigma1, mu2, sigma2
            All are the result of the ML method
    '''
    
    res = optimize.minimize(optimization_function, np.array([tau, mu1, sigma1, mu2, sigma2]),\
    args=x, tol=rtol, bounds = ((0, 1), (-np.inf, np.inf), (0, np.inf), (-np.inf, np.inf), (0, np.inf)))
    #С указанием границ, чтобы при плохой догадке не улететь в NaN
    #Тут опять эта подмена понятий, но ничего не могу поделать
    #Так написан optimize, res.x = [tau, mu1, sigma1, mu2, sigma2]
    #Итоговые приближения, естественно
    return res.x


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    '''
    EM method for a mixture of two normal distributions
    
    Parameters
    ----------
        x : np.array
            Contains values of x
        tau : float
            Relative number of objects in the first distribution
        mu1 : float
            Average position of the first distribution
        sigma1 : float
            The scatter of x in the first distribution
        mu2 : float
            Average position of the second distribution
        sigma2 : float
            The scatter of x in the second distribution
        rtol : float
            The relative tolerance parameter
            
    Returns
    -------
        tau, mu1, sigma1, mu2, sigma2
            All are the result of the EM method
    '''
    
    new = tau, mu1, sigma1, mu2, sigma2
    while 1:
        old = new
        res, d1, d2 = density_function(old, x)
        d1 = np.divide(tau * d1, res, where=d1!=0, out=np.full_like(d1, 0.5))
        d2 = np.divide((1 - tau) * d2, res, where=d2!=0, out=np.full_like(d2, 0.5))
        tau = (np.sum(d1) / x.size)
        mu1 = (np.sum(d1 * x) / np.sum(d1))
        sigma1 = (np.sqrt((np.sum(d1 * (x - mu1) ** 2)) / np.sum(d1)))
        mu2 = (np.sum(d2 * x) /np.sum(d2))
        sigma2 = (np.sqrt(np.sum(d2 * (x - mu2) ** 2) / np.sum(d2)))
        new = tau, mu1, sigma1, mu2, sigma2
        if np.allclose(new, old, rtol=rtol, atol=0):
            return np.asarray(new)


def density_function_d(args, x):
    '''
    Density functions for 3 multivariate normal distributions
    
    Parameters
    ----------
        args : object
            Contains all the neccesary data, as stated below:
        tau1, tau2 : float
            Relative numbers of stars in the first and second clusters
        muv : np.array with shape (2,)
            Mean proper motion of the cluster
        mu1 : np.array with shape (2,)
            Average position of the first cluster stars
        mu2 : np.array with shape (2,)
            Average position of the second cluster stars
        sigma02 : float
            The scatter of the proper motions of the stars of the field
        sigmax2 : float
            The scatter of the positions of cluster stars around the mean
        sigmav2 : float
            The scatter of proper motions of cluster stars relative to the mean
        rtol : float
            The relative tolerance parameter
            
    Returns
    -------
        res : np.array
            Sum of all density functions, multiplied by their weights
        density_1 : np.array
            Density functions values for the first distribution (normalized)
        density_2 : np.array
            Density functions values for the second distribution (normalized)
        density_3 : np.array
            Density functions values for the third distribution (normalized)
    '''
        
    tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2 = args
    density_1 = stats.multivariate_normal.pdf(x, mean=[*mu1, *muv], \
                    cov=np.diag([sigmax2, sigmax2, sigmav2, sigmav2]))
    density_2 = stats.multivariate_normal.pdf(x, mean=[*mu2, *muv], \
                    cov=np.diag([sigmax2, sigmax2, sigmav2, sigmav2]))
    density_3 = stats.multivariate_normal.pdf(x[:, 2:], mean=None,  \
                    cov=np.diag([sigma02, sigma02]))
    res = tau1 * density_1 + tau2 * density_2 + (1 - tau1 - tau2) * density_3
    density_1 = np.divide(tau1 * density_1, res, where=density_1!=0,\
                          out=np.full_like(density_1, 0.33))
    density_2 = np.divide(tau2 * density_2, res, where=density_2!=0,\
                          out=np.full_like(density_2, 0.33))
    density_3 = np.divide((1 - tau1 - tau2) * density_3, res, where=density_3!=0,\
                          out=np.full_like(density_3, 0.34))
    return res, density_1, density_2, density_3
   
def em_double_cluster(data, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-5, max_iter=100):
    '''
    EM method for a mixture of three normal distributions (double cluster solution)
    
    Parameters
    ----------
        data : np.array with shape (N, 4)
            Contains information about coordinates and speeds of stars
        tau1, tau2 : float
            Relative numbers of stars in the first and second clusters
        muv : np.array with shape (2,)
            Mean proper motion of the cluster
        mu1 : np.array with shape (2,)
            Average position of the first cluster stars
        mu2 : np.array with shape (2,)
            Average position of the second cluster stars
        sigma02 : float
            The scatter of the proper motions of the stars of the field
        sigmax2 : float
            The scatter of the positions of cluster stars around the mean
        sigmav2 : float
            The scatter of proper motions of cluster stars relative to the mean
        rtol : float
            The relative tolerance parameter
        max_iter : int
            Maxinun nuber of iterations
            
    Returns
    -------
        tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2
            All are the result of the EM method
    '''
    
    def comparison(new, old):
        tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2 = new
        mu1, mu2, muv = np.sqrt(mu1[0] ** 2 + mu1[1] ** 2),\
                        np.sqrt(mu2[0] ** 2 + mu2[1] ** 2),\
                        np.sqrt(muv[0] ** 2 + muv[1] ** 2)
        object1 = tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2
        tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2 = old
        mu1, mu2, muv = np.sqrt(mu1[0] ** 2 + mu1[1] ** 2),\
                        np.sqrt(mu2[0] ** 2 + mu2[1] ** 2),\
                        np.sqrt(muv[0] ** 2 + muv[1] ** 2)
        object2 = tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2
        return object1, object2
    
    new = tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2
    
    i = 0
    #Я вот даже уже со статьёй умных людей из гугла сравнил
    #Я вот уже миллион вещей сделал
    #Я вот уверен в своей реализации, но оно не работает :(
    while i < max_iter:
        old = new
        res, d1, d2, d3 = density_function_d(old, data)
        tau1, tau2 = np.mean(d1), np.mean(d2)

        mu1 = np.asarray([np.sum(d1 * data[:, 0]) / np.sum(d1), np.sum(d1 * data[:, 1]) / np.sum(d1)])
        mu2 = np.asarray([np.sum(d2 * data[:, 0]) / np.sum(d2), np.sum(d2 * data[:, 1]) / np.sum(d2)])
        muv = np.asarray([np.sum(d1 * data[:, 2]) / np.sum(d1), np.sum(d1 * data[:, 3]) / np.sum(d1)])
        
        sigmax2 = np.sum(d1 * (data[:,0] - mu1[0])**2 + d1 * (data[:,1] - mu1[1])**2 \
                         + d2 * (data[:,0] - mu2[0])**2 + d2 * (data[:,1] - mu2[1])**2) / (2 * np.sum(d1 + d2)) 
        sigmav2 = np.sum(d1 * ((data[:,2] - muv[0])**2 + (data[:,3] - muv[1])**2) \
                  + d2 * ((data[:,2] - muv[0])**2 + (data[:,3] - muv[1])**2)) / (2 * np.sum(d1 + d2)) 
        sigma02 = np.mean([np.sqrt(np.sum(d3 * (data[:, 2])**2) / np.sum(d3)),  np.sqrt(np.sum(d3 * (data[:, 3])**2) / np.sum(d3))])
        
        new = tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2

        if np.allclose(*comparison(new, old), rtol=rtol, atol=0):
            return new
        i += 1
        #print("iter:", i)
        #print("tau:", tau1, tau2)
        #print("mu:", mu1, mu2, muv)
        #print("sigma:", sigmax2, sigmav2, sigma02)
    return new

if __name__ == "__main__":
    pass
