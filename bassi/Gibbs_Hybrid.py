import numpy as np
from scipy.stats import truncnorm
from math import ceil
from time import time, localtime, strftime
from alive_progress import alive_bar

from .Functions import normalize_mat,CovCor, ABC, logP
from .Functions import weight_fun2,weight,Gen_len


############################################################
####### Truncated Multivariate Normal Distribution  ########
############################################################

def TMN(m, a, b, x0, mu, BurnIn = 0, sigma = np.array([None]), A = np.array([None]), saveoutput = True, nsave = 1, itera = 1, path=""):
    """
    Function to simulate from the multivariate truncated normal distribution.
    Directional Gibbs Sampler
    Combine the Mario propose (eigenvectors) and the marginal mutal information.

    Parameters
    ----------
    m  : int
        Sample size: Number of simulations
    a,b : tuple, list, or ndarray, optional
        Support of the distribution a < x < b.
    x0: array
        Initial state
    mu : tuple, list, or ndarray, optional
        Means vector.
    sigma : array
        Covariance matrix.
    A : array
        Precision matrix, inverse of the covariance matrix, you can give both sigma and A.
    saveoutput: bool
        If True save all simulations, otherwise only save the last state.
    nsave: int
        How many simulations do you want to save
    itera: int
        Iteration number
    path: str
        Directory where the output will be saved
    Returns
    -------
    simu, logEnergy: tuple
        If saveoutput = True
            Save m simulation from de MTN distribution (Simu)
            Save the log-posterior (logEnergy)
        else:
            return simu, logEnergy without save
    """

    sec = time()
    print("TruncMulNorm: Running the MCMC with %d iterations." % (m,), strftime("%a, %d %b %Y, %H:%M.", localtime(sec)))

    # Path to save the simulations simu and the logEnergy
    outname = path + "sim%sm%s.txt" % (itera, m)
    logname = path + "LogPos%sm%s.txt" % (itera, m)
    # Dimension
    d = len(mu)

    # The precision matrix and the covariance matrix are both required.
    if (A == None).any():
        A = np.linalg.inv(sigma)  # precision matrix.
    if (sigma == None).any():
        sigma = np.linalg.inv(A)  # Covariance matrix

    # Normalize the column of the covariance matrix (will be the directions).
    direc_C = normalize_mat(sigma, d)
    direc_C_T = direc_C.T.copy()
    
    sigma = CovCor(sigma)      # Correlation matrix (for the weight) equal to sigma to save memory.
    Inf_M = weight_fun2(sigma) # Probabilities of selections of directions
    del sigma  # to save memory

    #  We obtain the eigenvectors of the precision matrix (will be the directions).
    eigens = np.linalg.eigh(A)
    values = 1/eigens[0]
    
    direc_M = eigens[1]
    direc_M_T = direc_M.T.copy()
    del eigens
   
    eAe_C = np.diag(ABC(direc_C_T, A, direc_C))
    eAe_M = np.diag(ABC(direc_M_T, A, direc_M))

    del direc_C_T, direc_M_T

    # How many simulations do you want to save
    n_save = ceil(m / nsave) + 1
    ############################################################
    ############ Algoritmo Gibbs direcional optimo. ############
    ############################################################
    # MAP = x0
    # logP_MAP = logP(MAP, mu, A)
    yt = x0  # Initial state.

    simu = np.empty((n_save,d));  simu[0,:] = yt  # we save by row
    logEnergy = np.empty((n_save)); logEnergy[0] = logP(yt, mu, A)
    
    ### send an estimation for the duration of the sampling if
    sec2 = time()  # last time we sent a message

    i_save = 0
    with alive_bar(m) as bar: 
        for it in range(m):
            u = np.random.uniform() # To  choose the kernel (Mario or Cricelio)
            if u < 0.5: # This value 0.5 can be changed, a value between 0 and 1
                weigh = weight(values)  # Mario's proposal
                # Generate the new point yt from current point Dt
                a_r, b_r, mur, sigmar, e = Gen_len(yt, mu, A, direc_M, weigh, eAe_M, a, b)
            else:
                weigh_C = weight(Inf_M)
                a_r, b_r, mur, sigmar, e = Gen_len(yt, mu, A, direc_C, weigh_C, eAe_C, a, b)
            
            # Step length
            r2 = truncnorm.rvs(a=a_r, b=b_r, size=1)
            r = r2 * sigmar + mur  
            # Generates the new point yt
            yt += r * e
            
            # Compute the map (maximun between all simulations)
            # logP_yt = logP(yt, mu, A)
            # if logP_yt > logP_MAP:
            #     MAP = yt

            if ((it % nsave) == 0):
                i_save += 1
                simu[i_save, :] = yt                  # Save the current state
                logEnergy[i_save] = logP(yt, mu, A)   # Save log-posterior
                #ax = time()                          # Current time in iteration it
                #print("MTN: %7d/%s iterations so far. " % (it + 1, m) + Remain(m, it + 1, sec2, ax)) # We sent a message for remain time
                itm = it/m
                #print("[" + "-"*int(itm*69) + ">] " + str(np.round(itm*100,2)) + "%")

            bar()
        #print("[" + "-"*69 + ">] 100%")
        print("MTN: finished, " + strftime("%a, %d %b %Y, %H:%M:%S.", localtime(time())))
        Ttime = time() - sec2
        print("Finished in approx. %d minutes and %d seconds." % (Ttime // 60, Ttime % 60))

    # Save the last state: will be the initial state in next iteration
    np.savetxt(path + "x0%s.txt" % (itera), yt)  

    simu = simu[BurnIn//nsave:,:]
    logEnergy = logEnergy[BurnIn//nsave:]
    if saveoutput:
        with open(outname, 'w') as outfile:
            np.savetxt(outfile, simu)
        outfile.close()
        with open(logname, 'w') as outlogpos:
            np.savetxt(outlogpos, np.array([logEnergy]))
        outlogpos.close()
        return simu,logEnergy
    else:
        return simu, logEnergy#, MAP