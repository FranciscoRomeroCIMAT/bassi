import numpy as np
from scipy.linalg import solve
import os
import numba
from time import localtime, strftime

def pd_inv(A):
    """Compute the inverse of a positive definite matrix"""
    n = A.shape[0]
    I = np.identity(n)
    return solve(A, I, sym_pos = True, overwrite_b = True)

# =============================================================================
# Multiple matrix product
# =============================================================================
 
def ABC(A, B, C):
    return np.dot(np.dot(A, B), C)

# =============================================================================
# Normalize a vector
# =============================================================================
 
def normalize(v): return v/np.linalg.norm(v)
# =============================================================================
# Normalize the column of a matrix Mat of dimension d
# =============================================================================
#@jit(nopython=True)
def normalize_mat(Mat, d):
    SS = np.empty((d, d))
    for ij in range(d):
        SS[:, ij] = normalize(Mat[:, ij])
    return SS

# =============================================================================
# Log-posterior of a multivariate normal distribution yt~N(mu,A)
# =============================================================================
 
def logP(yt, mu, A):
    ymu = yt-mu
    sal = -0.5*ABC(ymu,A,ymu)
    return sal

# =============================================================================
# Create the correlation matrix from covariance matrix
# =============================================================================
 
def CovCor(sigma):
    D_inv = np.diag(1/np.sqrt(np.diag(sigma)))
    C = ABC(D_inv, sigma, D_inv)
    return C

# =============================================================================
# Probabilitys of selecting directions via the Mutual information marginal
# =============================================================================
epsilon = 1e-10  # To avoid numeric error

#@jit(nopython=True)
def weight_fun2(Corr): 
    Infmutua = - 0.5 * np.sum(np.log(np.power(Corr, 2) + epsilon), axis=1) # Cricelio' proposal
    Inf_1 = 1 / Infmutua
    return Inf_1

 
def weight(values):  # Mario's proposal
    bb = np.random.beta(2., 9, size=1)
    bb1 = values ** bb
    return bb1 / np.sum(bb1)

# =============================================================================
# Generates a random sample from a given 1-D array of probabilities
# =============================================================================
 
def rand_sample(weighs):
    return np.nonzero(np.random.multinomial(1, weighs))[0][0]

# Generate the new point yt from current point Dt
 
def Gen_len(Dt, mu, A_p, direc, weigh, eAe, a, b):
    ############################################################
    # i) We choose the direction
    ind = rand_sample(weigh)
    e = (direc[:, ind]).ravel()
    DTmu = Dt - mu
    ############################################################
    # ii) Generate the length r
    mur = - ABC(e, A_p, DTmu) / eAe[ind]  # Posterior mean
    sigmar = np.sqrt(1 / eAe[ind])  # Posterior standard deviation
    # Truncated support of r.
    c1 = (a - Dt) / e
    c2 = (b - Dt) / e
    c = np.max(np.concatenate((c1[e > 0], c2[e < 0])))
    db = np.min(np.concatenate((c1[e < 0], c2[e > 0])))
    a_r = (c - mur) / sigmar
    b_r = (db - mur) / sigmar

    return a_r, b_r, mur, sigmar, e

def Remain(Tr, it, sec1, sec2):
    """
    Remaining time Information messages:
    total iterations Tr, current it, start time, current time, as returned by time() (floats).
    """
    # how many seconds remaining
    n_bar = (70*it)//Tr
    porc = round(it*100/Tr,2)
    print("[%s>%s] " % ("-"*n_bar, " "*(70-n_bar-1))+str(porc)+"%")
    
    ax = int((Tr - it)*((sec2 - sec1)/it))
    if ax < 1:
        return " "
    elif ax < 60:
        return "Finish in approx. %d seconds." % (ax,)
    elif ax <= 360:
        return "Finish in approx. %d minutes and %d seconds." % (ax // 60, ax % 60)
    else:#(ax > 360):
        ax += sec2  # current time plus seconds remaining=end time
        return "Finish by " + strftime("%a, %d %b %Y, %H:%M.", localtime(ax))

def makedir(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName,  " Created ")
    else:
        print("Directory ", dirName,  " already exists")
        
############################################################
#########  Genera matrices de covarianzas  #################
############################################################ 
def genCov(d,alpha):
    """ Create square symmetric covariance matrix
    Requiere funciones: identity, random, qr, diag, inv, matmul
    
    Parameters
    ----------
    d  : int
        Dimention.
    alpha: float
        Correlation level (0  = independence)
    seed: float
        Fija la semilla.
    
    Returns
    -------
    Positive definite matrix (Covariance matrix).
    """
    seed = 23
    np.random.seed(seed)
    if alpha == 0:
        sigma = np.identity(d)
        np.random.seed()
        return(sigma)
    else:
        np.random.seed(seed)
        mat_unif = np.random.uniform(size=(d,d)) # Matriz with random antries uniform(0,1)
        q, r     = np.linalg.qr(mat_unif,mode="complete")  # Compute the qr factorization of a matrix.
        lamb     = np.diag(np.array(range(1,d+1)) ** (2*alpha/d))
        sigma    = np.linalg.inv(np.dot(np.dot(q.T,lamb),q)) # inv(T(q) * lamb * q)          
        np.random.seed()
        np.random.seed()
        return sigma


# Matern Autocovariance function

def rho(d, length):
    """
    The Matern covariance between two points separated by d distance.
    https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
    v = 3/2; p=1; sigma = 1

    Parameters
    ----------
    d: float
        Distance
    length: float
        Autocorreltion length
    """
    a = np.sqrt(3)
    return (1 + a*d/length)*np.exp(-a*d/length)


#### (0,0) cell is the SW corner subfault
def Index(i, j, ni):
    """Index of subfault (i,j) in the flattened array."""
    return i + ni*j


def Index_p(p, ni):
    """Index of subfault (i,j) in the flattened array."""
    return p[0] + ni*p[1]

def Loc(k, ni):
    """Location (i,j) of subfault with index k in the flattened array."""
    return (k % ni, k//ni)
### Index(Loc(23)[0], Loc(23)[1]) = 23, Loc(Index( 4, 1)) = (4,1)

def AutoCorr( p1, p2, fsize, length1, length2):
    """Autocorrelation between subfaults p1 and p2, array of indices
       Subfault sizes (fsize) and Autocorrelation length (length) in Km."""
    d = np.sqrt(np.sum((p1*fsize - p2*fsize)**2))  # Distance of p1 and p2 in Km
    return [rho(d, length=length1), rho(d, length=length2)]

def AutoCov(p1, p2, ni, nj,  fsize, length1, length2, Sigma):
    autocorr_12 = AutoCorr(p1, p2, fsize=fsize, length1=length1, length2=length2)
    S12 = Sigma[Index_p(p1,ni)] * Sigma[Index_p(p2,ni)]
    return S12 * np.array([autocorr_12[0], autocorr_12[1]])  # d_s and_d_d have zero correlation

def MakeVarCovMat(length1,length2,ni,nj,si,sj,Sigma):
   fsize=np.array([si, sj])
   d = 2*ni*nj
   VarCovMat = np.zeros((d, d))
   for ii in numba.prange(ni*nj):
       ii2 = 2*ii
       for jj in numba.prange(ni*nj):
           jj2 = 2*jj
           #AC12 = AutoCov(np.array(Loc(ii//2)), np.array(Loc(jj//2)), fsize, length1, length2, Sigma=1.0/Single_Wm)
           AC12 = AutoCov(np.array(Loc(ii,ni)), np.array(Loc(jj,ni)), ni, nj, fsize, length1, length2, Sigma)
           VarCovMat[ii2, jj2] = AC12[0]
           VarCovMat[ii2+1, jj2+1] = AC12[1]
   #VarCovMat += np.diag(np.array([1e-10] * d))
   return VarCovMat

def MakeAutoCorr(length1,length2,ni,nj,si,sj):
   fsize=np.array([si, sj])
   d = 2*ni*nj
   AutoCorrMat = np.zeros((d, d))
   for ii in numba.prange(ni*nj):
       ii2 = 2*ii
       for jj in numba.prange(ni*nj):
           jj2= 2*jj
           autocorr_12 = AutoCorr(np.array(Loc(ii)), np.array(Loc(jj)),  fsize=fsize, length1=length1, length2=length2)
           AutoCorrMat[ii2, jj2] = autocorr_12[0]
           AutoCorrMat[ii2+1, jj2+1] = autocorr_12[1]
   return AutoCorrMat

def Create_S0_A0(la1, la2, ni, nj, si, sj, beta1, beta2, W_m, border = True ):
    """
    Create the prior covarianza and precision matrix
    
    Parameters
    ----------
    la1 : float
        Correlation length for the strike component.
    la2 : float
        Correlation length for the dip component.
    border:
        Note: The variances in the lower and left border are modified (reduced) since no slip is expected in these areas.


    Returns
    -------
    S0 : array
        Covarianza matrix of the prior distribution.
    A0 : array
        Precision  matrix of the prior distribution.

    """
    d=2*ni*nj
    mm2 = 2*ni
    # left 
    left1 = np.arange(mm2) # dip and strike
    left2 = left1 + 2*ni; left2= left2[2:]
    left3 = left1+4*ni; left3= left3[4:]
    left4 = left1+6*ni
    # botom
    id1 = np.arange(0, d, mm2)
    bottom1 = np.append(id1,id1+1)
    bottom2 = np.append(id1[1:]+2,id1[1:]+3)
    bottom3 = np.append(id1[2:]+4,id1[2:]+5)

    bl_1 = np.unique(np.append(left1,bottom1))
    bl_2 = np.unique(np.append(left2,bottom2))
    bl_3 = np.unique(np.append(left3,bottom3))
    Single_Wm=W_m.flatten()
    #  Matern covariance function multyple by the weights W: WCW
    WCW = MakeVarCovMat(la1, la2, ni, nj, si, sj, Sigma=1/Single_Wm)
    # Along dip variance five times greater than along strike var.
    sig_vec = np.diag(np.array([1./beta1, 1./beta2]))
    SIG_mat = np.kron(np.identity(d // 2), sig_vec)
    if border:
        # To reduce de variance on the left and bottom border
        SIG_mat[bl_1,bl_1] /= 2.2
        SIG_mat[bl_2,bl_2] /= 1.5
        SIG_mat[bl_3,bl_3] /= 1.2
    
    S0 = ABC(SIG_mat, WCW, SIG_mat)  # Prior Covariance matrix
    A0 = pd_inv(S0)                  # Prior Precision maatrix
    return S0,A0 

def log_p_sig(x, alpha, beta):
    """The inverse gamma distribution's log probability density function
    with shape parameter alpha and scale parameter beta"""
    return (alpha+1) * np.log(x) + beta/x

def obj_fun_sig(x,*args):
    """
    Marginal posterior distribution of sigma^2*beta
    Parameters
    ----------
    x : float
        Beta parameter.

    Returns
    -------
    TYPE
        See Eq. 12 in Appendix B: Determining the Variance sigma2_beta.

    """
    sigma_2 = 1     # Residual variance fixed in 1
    sigma_m = x     # sigma2_beta in Eq. 7
    FS_0F = args[0]
    par_ = args[1]  # shape and scale parameter for sigma2_beta
    Sigma = args[2] # Covariance matrix of the data
    U0 = args[3]    #  Observations
    S_py = sigma_2*Sigma + sigma_m*FS_0F
    log_det_S_py = np.log(np.linalg.det(S_py))
    S_py_inv = np.linalg.inv(S_py)
    ymu = ABC(U0.T, S_py_inv, U0)
    return (log_det_S_py + ymu + log_p_sig(sigma_m, par_[0], par_[1]))

def alpha_beta(A0,T,U0,nobs):
    """ sig_beta^2 ~ Inv_Gam(v_m/2 , Sm/2) with shape and scale hyperparameters,
    v_m = 5 and Sm = Var(Y)xR2x(V_m+2) where the proportion of total variance R2
    was set to 0.75 for the beta coefficients and Var(Y) is the variance
    of the response variable.

    Parameters
    ----------
    A0 : array
        Prior precisio atrix.
    T : array
        Discretized FM operator built with the subfault tractions and areas.
    U0 : array
        Data.

    Returns
    -------
    v_m/2 and Sm/2: The hyperparameters for the prior distribition.

    """
    #nobs = 45  # Number of observations
    TTA0 = (np.diag(A0)*T).T
    TA0T = T@TTA0
    TrTA0T_n = np.trace(TA0T)/nobs
    
    #  Hyper-parameters
    Vy = np.var(U0) # Assume independent observations with homegenous variance
    R1 = 0.75       # Portion of the total variance Vy;
    v_m = 5         
    
    S_m = R1*Vy*(v_m + 2)/TrTA0T_n  
    alpha_m = v_m/2  # Shape parameter for sigma_beta
    beta_m = S_m/2   # Scale parameter for sigma_beta
    #sig2_beta = beta_m / (alpha_m+1) # prior mean
        
    return np.array([alpha_m, beta_m])
def Create_Covariance( nsli, rotang, Sn, Se, Sv):
    """ Create the square symmetric covariance matrix and the precision matrix A
    Likelihood U|D ~ N(TD,sigma^2 * S); A = inv(S), Sn, Se,Sv are standard deviations
    in metres in North, East and Vertical direction. nsli is number of GPS stations and
    rotang is Fault angle with respect of East Direction
    """
    
    # Variances
    Var_e = Se**2.;# North  
    Var_n = Sn**2.;# East  
    Var_v = Sv**2.;# Vertical  
    sigma2 = np.array([Var_n, Var_e, Var_v]);

    # Since this is the variance of one GPS measurement and we need two, so
    # to compute the displacement we do ??
    sigma2 *=2 # is necesary???
    
    # Rotation matrix
    R = np.array([[np.cos(np.radians(rotang)),np.sin(np.radians(rotang)),  0],
                   [-np.sin(np.radians(rotang)),np.cos(np.radians(rotang)),  0],
                   [0, 0, 1,]])
    
    # Variances aligned to north-east up. 
    gamma = np.diag(np.dot(R,sigma2))

    # gamma = R @ np.diag(sigma2) 
    
    # See section Data Likelihood in the paper
    Sigma = np.kron(np.identity(nsli), gamma) # Var-Cov matrix
    A = np.diag(1/np.diag(Sigma))             # Precision matrix = np.linalg.inv(Sigma) 
    
    return Sigma, A
def rotate_vector(U,rotang, z_invert=False):
    """Rotates 3d vector U rotang degrees in plane xy.
    If z_invert=True, changes z direction"""
    sign=1
    if (z_invert):
        sign=-1
    R = np.array([[np.cos(np.radians(rotang)),-np.sin(np.radians(rotang)),  0],
                   [np.sin(np.radians(rotang)),np.cos(np.radians(rotang)),  0],
                   [0, 0, sign,]])
    U_rot=U@R.T
    return(U_rot)

def Mag_Dis(x):
    return np.sqrt(x[::2]**2 + x[1::2]**2)