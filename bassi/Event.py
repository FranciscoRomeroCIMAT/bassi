import pandas as pd
import numpy as np
import os
from .Functions import makedir, Create_S0_A0, Create_Covariance
class event:
    """
        Fault. Fault object describing the Fault configuration of the Event.
        name: name of the Event, must coincide with folder name containing subsequent 
        Event data, and Analysis folders.
        stats_file: filename containing station data such as names and coordinates.
        file must be inside Event folder. Stat data is read as a pandas.DataFrame.
        U_file: filename of observation data in Event Folder. If not 
        specified can be later redefined which is useful in case of synthetic data.
        NE_direction: True if observations xy coordinates are North and East Directions, 
        and False if observations are along fault direction.
        Z_positive: True if Z coordinates (depth) are considered positive, false if 
        considered negative.
        T_file: name of file containing Traction Matrix in Event Folder.
        """
    def __init__(self, Fault, name, stats_file="selected_GPS_stats.dat", 
                 U_file=None, NE_direction=False, Z_positive=False, T_file="T.txt"):
        self.Fault=Fault
        self.name=name
        self.wkdir=Fault.rootdir+Fault.name+"/"+name+"/"
        try:
            self.stats=pd.read_table(self.wkdir+stats_file, sep="\s+",header=None)
        except:
            print("\t Stations data was not defined")
        
        self.nsli=len(self.stats)
        self.U=None
        try:
            self.U=np.loadtxt(self.wkdir+U_file)
        except:
            print("\t Warning: Observations not defined")
        self.NE_direction=NE_direction
        self.Z_positive=Z_positive
        self.T=np.loadtxt(self.wkdir+T_file)
        
    def U_fromfile(self, filedir, NE_direction=False, Z_positive=False):
        """Redefines observations U matrix from filedir. Must be Specified if observations
        are along NE_direction and if Z observations are positive. 
        Parameters
        ----------
        filedir:str
            directory of file from which to read observations data.
        NE_direction:bool
                True if data xy coordinates are in North and East directions. If
                false, data xy coordinates are assumed to be along Fault direction.
        Z_positive:bool
            True if data z coordinate (depth) is considered positive. If
            false, data z coordinates is considered negative.
        Returns
        ----------
        None."""
        self.U=np.loadtxt(filedir)
        self.NE_direction=NE_direction
        self.Z_positive=Z_positive
        
    def U_rot(self):
        """Rotates (and redefines) data matrix  along fault direction.
        Parameters
        ----------
        None.
        Returns
        ----------
        None.
        """
        try:
            U_rot=self.U
            rotang=self.Fault.rot_ang
            Z_positive=self.Z_positive
            sign=1
            if (Z_positive):
                sign=-1
            R = np.array([[np.cos(np.radians(rotang)),-np.sin(np.radians(rotang)),  0],
                           [np.sin(np.radians(rotang)),np.cos(np.radians(rotang)),  0],
                           [0, 0, sign,]])
            U_rot=U_rot@R.T
            return(U_rot)
        except:
            print("U observations not defined")
    def create_synthetic(self, la1, la2, sig2_bet, mode1, Sn, Se, Sv, beta1=1., beta2=0.2,
                          two_mode = False, mode2=None, 
                          W_file="W_m.txt"):
        
        """
        Simulate synthetic data with given parameters.
        Parameters
        ----------
        la1 : float
            Correlation length for the strike component.
        la2 : float
            Correlation length for the dip component.
        sig2_bet : float
            variance of the prior distribution; sig_beta^2.
        mode1: int
            index of subfault which would contain the highest slip magnitude
        Sn: float
            standard deviation of displacements in North Direction
        Se: float
            standard deviation of displacements in East Direction
        Sv: float
            standard deviation of displacements in Vertical Direction
        beta1:float
            along strike precision for prior precision Matrix
        beta2:float
            along dip precision for prior precision Matrix
        two_mode : bolean, optional
            If two_mode = True simulate the synthetic data with two modes. The default is False.
        W_file:str
            filename of Weight Matrix. Must be contained in Fault Folder.
        Returns
        -------
        None.

        """
        path = self.Fault.rootdir
        path_Fault = path+self.Fault.name+"/"
        path_Event=self.wkdir
        path_la = path_Event+ "Sim_la1_%s_la2_%s/" % (la1, la2)
        path_synthetic = path_Event + "Synthetic_la1_%s_la2_%s/" % (la1, la2)
            
        # Create the path for the synthetic data
        makedir(path_synthetic)
        makedir(path_la)
        
        # # To use the real cases to faound the maximun splip
        # simu_goog = np.loadtxt(path_synthetic + "sim1m1570000.txt")  
        # des = np.quantile(simu_goog,0.5,axis=1)#sim_goog.mean(axis=1)
        # #des = np.loadtxt(path + "DC.out") # Map of la1 = 40 and la2 = 100
        # desx = des[::2]
        # # along dip components
        # desy = des[1::2]
        # # We use the median displacement, for la1 = 40 la2 = 45
        # Magnitud = np.sqrt(desx ** 2 + desy ** 2)/2
        ni=self.Fault.shape[0]
        nj=self.Fault.shape[1]
        si=self.Fault.sfsize[0]
        sj=self.Fault.sfsize[1]
        W_m=self.Fault.WeightMat(W_file=W_file)
        S0,A0 = Create_S0_A0(la1, la2, ni, nj, si, sj, beta1, beta2, W_m )
        A_0 = A0/sig2_bet
        S_0 = S0*sig2_bet
        # Save the prior parameters
        np.savetxt(path_la + "Ap.txt", A_0)
        np.savetxt(path_la + "Sp.txt", S_0)
        
        
        # Dimension
        d = 2*ni*nj
        mu_0 = np.repeat(0,d).reshape((d,1)) #  Prior mean
        # Index
        full_index = [i for i in range(d)]
        
        if two_mode: 
            ind_max = np.array([mode1,mode2])
            x_1 = np.array([0.0004, 0.2,-0.002,0.08]).reshape((4,1))
        
            ind_complete_max = 2*ind_max[0]
            ind_complete_2 = 2*ind_max[1]
            # Index of X_1
            index_xi = np.array([ind_complete_max,ind_complete_max+1,ind_complete_2,ind_complete_2+1])
        else:
            ind_max = mode1 #  np.argmax(Magnitud); whrere the maximin slip was found
            # We save only de max displacement X_i = x_i
            x_1 = np.array([0.0004, 0.2]).reshape((2,1))
            #x_1 = np.array([desx[ind_max], desy[ind_max]]).reshape((2,1))
            ind_complete = 2*ind_max # Index in the full vector
            index_xi = np.array([ind_complete,ind_complete+1]) # Index of X_i
        
        # We keep the remaining values; remove index_xi from full_index
        [full_index.remove(i) for i in index_xi]
        
        remain_index = np.array(full_index) # Full index without index_xi
        # X_i|X_{-i}~N(mu_0[remain_index] + np.dot(Sigma21_11, x_1 - mu_1),  Sigma22 + Sigma21_11@Sigma12)
        ind_cov = np.meshgrid(index_xi,index_xi)
        Sigma11 = S_0[ind_cov[0], ind_cov[1]]
        
        Sigma22_aux = S_0[remain_index, :]; #Sigma22 = Sigma22_aux[:, remain_index]
        Sigma21 = Sigma22_aux[:, index_xi]; #Sigma12 = Sigma21.T # Sigma12 = Sigma21_aux[index_xi, :]
        
        Sigma11_inv = np.linalg.inv(Sigma11)
        mu_1 = mu_0[index_xi]
        
        Sigma21_11 = Sigma21@Sigma11_inv
        # Parametert of X_2 | X_1 = x_1
        mu2 = mu_0[remain_index] + np.dot(Sigma21_11, x_1 - mu_1)
        #Sigma2 = Sigma22 + Sigma21_11@Sigma12; #A2 = np.linalg.inv(Sigma2)
        
        sim = np.repeat(0., d)
        sim[index_xi] = x_1.ravel()
        sim[remain_index] = mu2.ravel() # The mean is in the sport!
        
        # # =============================================================================
        # #  Synthetic observations
        # # =============================================================================
        T = self.T
        # =============================================================================
        # Likelihood U|D ~ N(TD,sigma^2 * S); A = inv(S)
        # =============================================================================
        Sigma, A = Create_Covariance(nsli = self.nsli,
        rotang = self.Fault.rot_ang, Sn=Sn, Se=Se, Sv=Sv)
        TD = np.dot(T, sim)
        # Synthetic displacement without noise
        from scipy.stats import multivariate_normal
        u_sim = multivariate_normal.rvs(mean=TD,cov=Sigma)
        
        self.U_synt=u_sim
        # Save the synthetic data
        # Synthetic slip
        np.savetxt(path_synthetic + "sim_la1_%s_la2_%s.out" % (la1, la2),sim.T) # Slip
        # Synthetic displacements without noise
        np.savetxt(path_synthetic + "u_sim_la1_%s_la2_%s.out" % (la1, la2),TD)
        # Synthetic displacements with noise
        np.savetxt(path_synthetic + "u_sim_la1_%s_la2_%s_error.out" % (la1, la2),u_sim.T)
        """
        # =============================================================================
        #  Plot the synthetic slip
        # =============================================================================
         
        if plot_simulated:
            import matplotlib.pyplot as plt
            from PlotMapa import PlotInten
            PlotInten(sim, new=True, color="red", alpha=1)
            plt.savefig(path_synthetic + "Sinthetyc_D.png")
        #PlotInten(des, new=True,  color="red", alpha=1)
        #plt.savefig(path_synthetic + "Estimate_D.png")"""
    def stats_np(self,stats_name):
        """Returns stats data as numpy array. May not work if file read contains
        different types of data."""
        return(np.loadtxt(self.wkdir+"/"
                                  +stats_name))
        
    def stats_df(self,stats_name):
        """Returns stats data as pandas.Dataframe"""
        stations= pd.read_table(self.wkdir+stats_name
                                ,sep="\s+",header=None,index_col=False)
        return(stations)
    def reshape_U(self,nrow,ncol):
        """Reshapes U matrix."""
        self.U=np.reshape(self.U,(nrow,ncol))

    