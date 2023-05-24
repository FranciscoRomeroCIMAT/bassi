# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:18:09 2023

@author: Francisco Romero
"""
 
import bassi as ba
import os

path=os.getcwd()+"/"
GGap=ba.fault(name="Guerrero",
                   rot_ang=-21.1859,shape=(19,31),
                   sfsize=(20.563905505456397,20.002022023921285),
                   rootdir=path)

SSE_2006=ba.event(GGap, name="SSE_2006",
                           stats_file="stationsLL_2006.txt", 
             U_file="u.dat", NE_direction=False, Z_positive=False,
             T_file="Tractions_SSE2006_Radiguet_x_subfaults_areas.in")


Test_analysis=ba.analysis(SSE_2006, folder_name="Test_analysis", Sn=0.0021, Se=0.0025, Sv=0.0051,
                               s_supp=(-0.1,0.1), d_supp=(-0.0804,0.4),beta1=1.,beta2=0.2 )
##### Real Data
Test_analysis.Post(la1=35, la2=45, synt = False, la1_s = None, la2_s = None, sig2_betar = None)

simu,logpost = Test_analysis.Run_MCMC(35, 45, itera=1, 
                    n = 50000, 
                    BurnIn = 6000, 
                    ss = 5000,  
                    saveoutput = True, saveconfig=True)
#Synthetic Data
SSE_2006.create_synthetic(la1=35, la2=45, sig2_bet=0.0004, 
                              mode1=254, Sn=0.0021, Se=0.0025, 
                              Sv=0.0051, beta1=1., beta2=0.2,
                      two_mode = False, mode2=None, 
                      W_file="W_m.txt")

Test_analysis.Post(la1=47, la2=39, synt = True, la1_s = 35, la2_s = 45, sig2_betar = None)

simu,logpost = Test_analysis.Run_MCMC(47, 39, itera=1, 
                    n = 50000, 
                    BurnIn = 6000, 
                    ss = 5000,  
                    saveoutput = True, saveconfig=True)

