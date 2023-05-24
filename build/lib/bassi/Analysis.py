import numpy as np
from .Functions import ABC, Create_S0_A0, alpha_beta,Mag_Dis, obj_fun_sig, makedir, Create_Covariance
from scipy.optimize import minimize
from .Gibbs_Hybrid import TMN
import ast

class analysis:
    """
        Event. Event object describing the Event data.
        Sn: Standard deviation in north direction
        Se: Standard deviation in east direction
        Sv: Standard deviation in vertical direction
        s_supp:tuple containing slip along strike direction support
        d_supp: tuple containing slip along dip direction support
        synt_data: True if data to analyze is synthetic, false if data to analyze is real
        beta1:  Precision for strike component, default is 1
        beta2: Precision for dip component, default is 0.2 
        """
    def __init__(self, Event, folder_name, Sn, Se, Sv, s_supp, d_supp, synt_data=False,  beta1=1.,beta2=0.2 ):
        self.Event=Event
        self.Sd=(Sn,Se,Sv)
        self.beta=(beta1,beta2)
        self.s_supp=s_supp
        self.d_supp=d_supp
        self.Fault=self.Event.Fault
        self.U=self.Event.U
        makedir(Event.wkdir+folder_name+"/")
        self.output_dir=Event.wkdir+folder_name+"/"
        if synt_data:
            self.U=self.Event.U_synt

    def logV(self,theta):
        """loglikelihood U|D ~ N(TD,S); A = inv(S). T and S are computed through Event object
        parameters.
        Parameters
        ----------
        theta: D vector in likelihood function
        Returns
        ----------
        likelihood evaluated at D=theta."""
        # =============================================================================
        # Likelihood U|D ~ N(TD,S); A = inv(S)
        # =============================================================================
        Sigma, A = Create_Covariance(nsli = self.Event.nsli,
        rotang = self.Fault.rot_ang, Sn=self.Sd[0], Se=self.Sd[1], Sv=self.Sd[2])
        U0 = self.Event.U
        U0=U0.flatten()# .reshape((nobs, 1)) # Data: len(U0) = nobs observations
        T = self.Event.T
        mu = np.dot(T, theta)
        ymu = U0 - mu
        return -0.5 * ABC(ymu, A, ymu)


    def logLike(self,theta, U0,T):
        """logLikelihood U|D ~ N(TD,S); A = inv(S).  S is computed through Event object
        parameters. T and U0 are read as parameters.
        Parameters
        ----------
        theta: D vector in likelihood function
        T: Traction Matrix
        U0:  O0bservations data. Must be a flatten array and directions should be
        along fault direction
        Returns
        ----------
        likelihood evaluated at D=theta."""
        Sigma, A = Create_Covariance(nsli = self.Event.nsli,
        rotang = self.Event.Fault.rot_ang, Sn=self.Sd[0], Se=self.Sd[1], Sv=self.Sd[2])
        mu = np.dot(T, theta)
        ymu = U0 - mu
        return -0.5 * ABC(ymu, A, ymu)
    
    def Post(self,la1, la2, synt = True, la1_s = None, la2_s = None, sig2_betar = None):
        """
        Function to compute the posterior mean and posterior covariance matrix of
        the multivariate truncated normal distribution.
        Parameters
        ----------
        la1 : float
            Correlation length for the strike component.
        la2 : float
            Correlation length for the dip component.
        synt : Bolean
            If synt = True uses the synthetic data.
            If synt = False uses the real data.
        la1_s : float
            Correlation length for the strike component for the synthetic data.
        la2_s : float
            Correlation length for the dip component for the synthetic data.    
        sig2_betar : float
            Variance in definition of prior Covariance Matrix
        Returns
        -------
            Posterior mean and posterior covariance matrix.
        """
        
        path = self.Event.Fault.rootdir
        path_Fault = path + self.Event.Fault.name +"/"             # Path with data
        path_Event=path_Fault+self.Event.name +"/"
        path_la = self.output_dir + "Sim_la1_%s_la2_%s/" % (la1,la2) # Path to save the mean and covariance
        path_la_sali = path_la + "output/"  # Path to save the initial state for future simulations

        # Discretized FM operator built with the subfault tractions and areas
        T = self.Event.T
        if synt: # Synthetic data
            path_la1 = path_Event + "Synthetic_la1_%s_la2_%s/" % (la1_s, la2_s)
            U0 = np.loadtxt(path_la1 + "u_sim_la1_%s_la2_%s.out" % (la1_s, la2_s))
        else: # Real data
             U0 = U0=self.Event.U # Data: len(U0) = nobs observations
        
        # =============================================================================
        # Likelihood U|D ~ N(FD, S); A = inv(S)
        # =============================================================================
        # Var-Cov matrix Sigma and Precision A = inv(Sigma) 
        Sigma, A = Create_Covariance(nsli = self.Event.nsli,
        rotang = self.Event.Fault.rot_ang, Sn=self.Sd[0], Se=self.Sd[1], Sv=self.Sd[2])
        
        
        # print('el valor de sigma es ' + str(Sigma))
        # =============================================================================
        # Priors D ~ N(mu,sig_beta^2 * S0); S0 = B⁻WCWB⁻ = inv(BWC⁻WB)
        # =============================================================================
        S0, A0 = Create_S0_A0(la1=la1, la2=la2,
                              ni=self.Event.Fault.rows(),nj=self.Event.Fault.col(),
                              si=self.Event.Fault.sfsize[0], sj=self.Event.Fault.sfsize[1],
                              W_m=self.Event.Fault.WeightMat(),
                              beta1=self.beta[0], beta2=self.beta[1]) # Prior Covariance and Precision matrix
        
        # =============================================================================
        # sig_beta^2 ~ Inv_Gam(v_m/2 , Sm/2)
        # Establishing the hyperparamete sig_beta^2
        # =============================================================================
        TT = T.T.copy()    
        TAT = ABC(TT, A, T)
        
        nobs=len(self.Event.U)
        Uflat=U0.flatten()
        #  Computing the hyperparameters for the prior distribition of sig_beta^2.
        par_pri = alpha_beta(A0,T,Uflat,nobs)
        TS_0T = ABC(T, S0, T.T)
        x01 = np.array([par_pri[1]/(par_pri[0]+1)])
        
        res = minimize(obj_fun_sig, x01, args=(TS_0T, par_pri, Sigma, Uflat), bounds=[(1e-12,1)])
        
        # Capella: Vamos a experimentar con una sig2_beta constante
        #          Cricelio la calcula aqui, pero su valos depende fuertemente 
        #          de la1 y la2 haciendose pequeña en la misma region que el dic     
        
        if ( sig2_betar == None ):
            sig2_beta = res.x[0] # 0.0004
        else:
            sig2_beta = sig2_betar

        print("Sigma^2_beta = %s" %sig2_beta)
        
        x = np.array([la1, la2, sig2_beta])
        outname_var = path_Event + "Variances.txt" # Save the variance estimated
        with open(outname_var, 'a') as outfile:
            np.savetxt(outfile, x.reshape((1,len(x))), fmt=['%-3.0f','%-3.0f','%-7.10f'])        
        outfile.close()
        # =============================================================================
        # Posterior D ~ N(mu_p, S_p). See page 9 of the paper.
        # =============================================================================
        
        TA=np.zeros(T.T.shape)
        TA = np.matmul(T.T, A)
        
        TAT = np.matmul(TA,T)
        
        
        A_p = TAT + A0/sig2_beta # Posterior precision
        S_p = np.linalg.inv(A_p) # Posterior Covariance

        TAUA0mu = TA @ Uflat  # + A0@mu # The second part is when the prior mean is different to 0
        Theta_p = S_p @ TAUA0mu  # Posterior mean

        # Create target Directory if don't exist
        makedir(path_la)
        # Save the results
        np.savetxt(path_la + "/S0.txt", S0) # Prior Covariance matrix
        np.savetxt(path_la + "/A0.txt", A0) # Prior precision matrix
        np.savetxt(path_la + "Ap.txt", A_p) # Posterior Covariance matrix
        np.savetxt(path_la + "Sp.txt", S_p) # Posterior Covariance matrix
        np.savetxt(path_la + "Thetap.txt", Theta_p) # Posterior mean 
        
        # =============================================================================
        #     Initial state value for the MCMC simulations
        # =============================================================================
        x0 = np.repeat(0.,len(Theta_p))
        x0[1::2] = 0.05 # Can be changed for another small value
        # Create target Directory if don't exist
        makedir(path_la_sali)
        np.savetxt(path_la_sali+"x00.txt",x0)
        return Theta_p
    
    def DIC(self,data):
        """Computes DIC of posterior simulated data
        ----------
        data: numpy.array
            simulated data
        ----------
        Deviance information criterion as described in Spiegelhalter et al. (2002, p. 587)."""
        theta_mean = data.mean(axis = 0)  #Posterior mean
        log_vero = np.apply_along_axis(self.logV, 1, data)
        DIC_all = - 2 * log_vero          # D(theta_vector)
        DIC_mean = - 2 * self.logV(theta_mean) # D(theta_mean)
        mean_DIC =  DIC_all.mean(axis = 0)    # mean(D(theta_vector))
        DIC_i = 2 * mean_DIC - DIC_mean
        return DIC_i
    
    def Run_MCMC(self, la1, la2, itera = 1, n = 15000000, ss = 2000, BurnIn = 60000, saveoutput = False, saveconfig=False): 
        """
        We simulate from the truncated normal calculating the DIC
        Parameters
        ----------
        la1 : float
            Correlation length for the strike component.
        la2 : float
            Correlation length for the dip component.
        itera : int
            Number of the iteration.
        n: int
            number of simulations: length of the chain without Burn-in
        ss: int
            Sample size; number of simulations to save.
            
        saveoutput : bolean, optional
            If saveoutput = True, save the simulations else only return the simulations
            simu and the log energy logEnergy. The default is False.
            
        saveconfig : bolean, optional
            If saveconfig = True, save the MCMC config in a file else the MCMC config is not
            saved. The default is False.

        Returns
        -------
        Save la1, la2, and the DIC in the DIC_all.txt file.

        """
        
        # paths the simulations and outputfiles
        path = self.Fault.rootdir
        path_Fault = path + self.Fault.name +"/"             # Path with data
        path_Event=path_Fault+self.Event.name +"/"
        path_la = self.output_dir + "Sim_la1_%s_la2_%s/" % (la1,la2) # Path to save the mean and covariance
        path_la_sali = path_la + "output/"  # Path to save the initial state for future simulations

        
        # Create target Directory if don't exist
        makedir(path_la_sali)
        
        # Posterior precision matrix and posterior mean  
        A_p = np.loadtxt(path_la + "Ap.txt" )
        S_p = np.loadtxt(path_la + "Sp.txt" )
        Theta_p = np.loadtxt(path_la + "Thetap.txt" )
        
        d = len(Theta_p)
        # We restrict the support a_s < d_s < b_s and a_d < d_d < b_s
        a_s = self.s_supp[0];  b_s = self.s_supp[1]
        a_d = self.d_supp[0]; b_d = self.d_supp[1];
        
        a = np.array([a_s, a_d] * (d // 2))
        b = np.array([b_s, b_d] * (d // 2))

        # With this cofiguation, this is a good burn-in period abouit 60000; 
        # see the log-posterior.

        m = BurnIn + n # Number of simulations
        x0 = np.loadtxt(path_la_sali +"x0%s.txt"%(itera-1))  # Give a initial value

        simu,logEnergy = TMN(m=m, a = a, b = b, x0 = x0, mu = Theta_p, 
                            BurnIn = BurnIn, sigma = S_p, A = A_p, 
                            saveoutput = saveoutput, 
                            nsave=m//ss, itera=itera, 
                            path = path_la_sali)

        # Save the DIC

        x = np.array([la1,la2 ,self.DIC(simu)]) 
        outname_DIC = path_la_sali+"DIC.txt"
        
        with open(outname_DIC, 'a') as outfile:
            np.savetxt(outfile, x.reshape((1,len(x))), 
                        fmt=['%-3.0f','%-3.0f','%-7.12f'])
        outfile.close()
        config=dict(n=n,itera=itera, ss=ss, BurnIn=BurnIn, saveoutput=saveoutput)
        if saveconfig:
            config=dict(n=n,itera=itera, ss=ss, BurnIn=BurnIn, saveoutput=saveoutput)
            f = open(path_la_sali+"config.txt","w")
            f.write( str(config) )
            f.close()
        return simu,logEnergy 
    
    def Compute_CV(self,la1,la2):
        """Computes Coefficient of Variation
        Parameters
        ----------
        la1 : float
            Correlation length for the strike component.
        la2 : float
            Correlation length for the dip component.
    
        Returns
        -------
        Saves Coefficient of Variaton and returns it as a numpy array"""
        path = self.Fault.rootdir
        path_Fault = path + self.Event.Fault.name +"/"             # Path with data
        path_Event=path_Fault+self.Event.name +"/"
        path_la = self.output_dir + "Sim_la1_%s_la2_%s/" % (la1,la2) # Path to save the mean and covariance
        path_la_sali = path_la + "output/"  # Path to save the initial state for future simulations
        
        with open(path_la_sali+'config.txt') as f:
            dictoniary = f.read()
        config = ast.literal_eval(dictoniary)
        m= config["BurnIn"] + config["n"]
        itera=config["itera"]
        data1 = np.loadtxt(path_la_sali+"sim%sm%s.txt" % (itera, m))

        Mag = np.apply_along_axis(Mag_Dis,1,data1)
        #Q25 = np.quantile(Mag,0.25,axis=0)
        Q50 = np.quantile(Mag,0.5,axis=0)
        #Q75 = np.quantile(Mag,0.75,axis=0)
        #IQR = Q75-Q25
        #CV = IQR/(Q50+8e-03)
        #CV =  std(Mag,axis=0)/(mean(Mag,axis=0)+8e-03)

        CV = np.std(Mag,axis=0)/(Q50+8e-03)

        np.savetxt(path_la_sali+"CV_%s_%s.txt"%(la1, la2),CV.T)
        return(CV)