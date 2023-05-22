import numpy as np
import pandas as pd
import os



class fault:
    """
        name: name of the fault, must coincide with folder name containing subsequent 
        Fault data, and Event folders.
        rot_ang: rotation angle of Fault with respect to the East Direction
        shape: shape of fault (number of rows, number of columns)
        sfsize: size of each subfault in Km (row width, column width)
        rootdir: root directory where Fault Folder is located.
        
    """
    def __init__( self, name, rot_ang, shape, sfsize=None,
                 rootdir=os.getcwd()+"/"):
        self.name=name
        self.rootdir=rootdir
        self.rot_ang=rot_ang
        self.shape=shape
        self.sfsize=sfsize
        self.dim=shape[0]*shape[1]
        try:
            self.row_width=sfsize[0]
            self.col_width=sfsize[1]
        except:
            #Should subfault size not be specified.
            print("\n Warning:Subfault size not initiliazed")
            
    def Sf_size_fromfile(self,filename="subfault_size.txt"):
        """Reads subfault size in Km from file. Used if not initialized or needs to be
        redefined.
        Parameters
        ----------
        filename:str
            name of file containing subfault sizes located in Fault folder
        -------
        None."""
        sizes=np.loadtxt(self.rootdir+self.name+"/"+filename)
        self.sfsize=sizes
        self.row_width=sizes[0]
        self.col_width=sizes[1]
        
    def WeightMat(self, W_file="W_m.txt"):
        """Returns Weight Matrix for Fault Config as Numpy array, file 
        must be located in fault folder or specified subfolder of fault folder.
        Parameters
        ----------
        W_file:str
            name or subdirectory of Weight Matrix File
        
        Returns
        -------
        W: numpy.array"""
        W=np.loadtxt(self.rootdir+self.name+"/"+W_file)
        return(W)

    def rows(self):
        """Returns number of rows of Subfault Configuration.
        Parameters
        ----------
        None
        
        Returns
        -------
        nrows:int"""
        return(self.shape[0])

    def col(self):
        """Returns number of columns of Subfault Configuration
        Parameters
        ----------
        None
        
        Returns
        -------
        ncols:int
        """
        return(self.shape[1])
    
    def sf_x_coords(self, filename="sfcoorx.txt"):
        """Returns Subfaults East coordinates as Pandas Dataframe. CRS not assigned
        Parameters.
        ----------
        filename:str
            name of file containing x coordinates of some CRS.
    
        Returns
        -------
        pandas.Dataframe"""
        return(pd.read_table(self.rootdir+self.name+"/"
                            + filename,sep="\s+",header=None,index_col=False))
    
    def sf_y_coords(self, filename="sfcoory.txt"):
        """Returns Subfaults North coordinates as Pandas Dataframe. CRS not assigned.
        Parameters.
        ----------
        filename:str
            name of file containing y coordinates of some CRS.
    
        Returns
        -------
        pandas.Dataframe"""
        return(pd.read_table(self.rootdir+self.name+"/"
                            + filename,sep="\s+",header=None,index_col=False))
    
    def sf_z_coords(self,filename="sfcoorz.txt"):
        """Returns Subfaults Vertical Depths as Pandas Dataframe. CRS not assigned.
        Parameters.
        ----------
        filename:str
            name of file containing z coordinates of some CRS.
    
        Returns
        -------
        pandas.Dataframe"""
        return(pd.read_table(self.rootdir+self.name+"/"
                            + filename,sep="\s+",header=None,index_col=False))
    

    
    