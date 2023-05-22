# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:18:09 2023

@author: Francisco Romero
"""
 
import bassi as ba
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt



GGap=ba.fault(name="Guerrero", loc="Guerrero",
                   rot_ang=-21.1859,shape=(19,31),
                   sfsize=(20.563905505456397,20.002022023921285),
                   rootdir="C:/Users\migue\OneDrive\Escritorio\Codes_SSE\Prueba/")

SSE_2006=ba.event(GGap, name="2006",
                           stats_file="stationsLL_2006.txt", 
             U_file="u.dat", NE_direction=False, Z_positive=False,
             T_file="Tractions_SSE2006_Radiguet_x_subfaults_areas.in")
SSE_2006.reshape_U(15, 3)
#print(Evento2006.U_rot())

Test_analysis=ba.analysis(SSE_2006, Sn=0.0021, Se=0.0025, Sv=0.0051,
                               s_supp=(-0.1,0.1), d_supp=(-0.0804,0.4),beta1=1.,beta2=0.2 )


Test_analysis.Post(la1=47, la2=39, synt = False, la1_s = None, la2_s = None, sig2_betar = None)

SSE_2006.create_synthetic(la1=35, la2=45, sig2_bet=0.0004, 
                              mode1=254, Sn=0.0021, Se=0.0025, 
                              Sv=0.0051, beta1=1., beta2=0.2,
                      two_mode = False, mode2=None, 
                      W_file="W_m.txt")

simu,logEnergy = Test_analysis.Run_MCMC(47, 39, itera=1, 
                    n = 500000, 
                    BurnIn = 60000, 
                    ss = 5000,  
                    saveoutput = True, saveconfig=True)



"""
sfcoorx=sfx.to_numpy()
sfcoory=sfy.to_numpy()
geometry=gpd.points_from_xy(sfcoorx.flatten(), sfcoory.flatten(),crs="epsg:6369")
df=gpd.GeoDataFrame(geometry=geometry,crs="epsg:6369")

df=df.to_crs("epsg:4326")

geometry=gpd.points_from_xy(sfcoorx.flatten(), sfcoory.flatten(),crs="epsg:6369")

mex=gpd.read_file("C:/Users/migue/Downloads/889463770541_s/mg2022_integrado/conjunto_de_datos/00ent.shp")
mex=mex.set_crs("epsg:6372",allow_override=True)
 
mex=mex.to_crs("epsg:4326")   

fig, ax = plt.subplots(figsize=(16, 16))
mex.plot(ax=ax,color='snow', edgecolor='gray')
rectangles=df.geometry.rotate(21.018)
rectangles=rectangles.buffer(0.043,cap_style=3)
rectangles=rectangles.rotate(-21.018)

grid=gpd.GeoDataFrame(df,geometry=rectangles,crs="epsg:4326")

grid.plot(ax=ax,alpha=0.6,legend=True,
          vmin=0,vmax=0.2,
        legend_kwds={'location':'bottom',
                     "shrink":0.5,
                     "label":"Slip Magnitude (m)",
                     "pad":0.04})

#gdf.plot(ax=ax,color="greenyellow",marker="^",edgecolor="black",markersize=300)
plt.xlim([-102.5,-96])
plt.ylim([15.5,20])
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")

stats=Window1.stats_df(stats_name="selected_GPS_stats.dat")

gdf = gpd.GeoDataFrame(
    stats, geometry=gpd.points_from_xy(stats[1], stats[2]))
gdf.plot(ax=ax,color="greenyellow",marker="^",edgecolor="black",markersize=300)

# %%
AnalisisWindow1=Analysis.Analysis(Window1, Sn=0.0021, Se=0.0025, Sv=0.0051,
                               s_supp=(-0.1,0.1), d_supp=(-0.0804,0.4),beta1=1.,beta2=0.2 )

la1=35
la2=45
n=200000
BurnIn=6000
ss=5000
m=n+BurnIn
itera=1
AnalisisWindow1.Post(la1=la1, la2=la2, synt = False, la1_s = None, la2_s = None, sig2_betar = 0.0004)
simu,logEnergy = AnalisisWindow1.Run_MCMC(la1, la2, itera=itera , 
                    n = n, 
                    BurnIn = BurnIn, 
                    ss = ss,  
                    saveoutput = True, saveconfig=True)
CV=AnalisisWindow1.Compute_CV(la1=la1,la2=la2)
salida=Window1.wkdir+"Sim_la1_%s_la2_%s\salida/"%(la1,la2)
simu=np.loadtxt(salida+"sim%sm%s.txt"%(itera,m))
logEnergy=np.loadtxt(salida+"LogPos%sm%s.txt"%(itera,m))
CV=np.loadtxt(salida+"CV_%s_%s.txt"%(la1,la2))
fig1, axs = plt.subplots(1,)                        
plt.plot(logEnergy)
plt.title("logEnergy la1_%s_la2_%s" % (la1, la2))

sfcoorx=sfx.to_numpy()
sfcoory=sfy.to_numpy()

di_tot=simu
di=np.median(di_tot,axis=0)
d=np.reshape(di,(len(di)//2,2))
slip=np.linalg.norm(d,axis=1)
slip=np.reshape(slip,FallaGuerrero.shape)

geometry=gpd.points_from_xy(sfcoorx.flatten(), sfcoory.flatten(),crs="epsg:6369")
df=gpd.GeoDataFrame(slip.flatten(),geometry=geometry,crs="epsg:6369")
df=df.assign(CV=CV)
df=df.to_crs("epsg:4326")

mex=gpd.read_file("C:/Users/migue/Downloads/889463770541_s/mg2022_integrado/conjunto_de_datos/00ent.shp")
mex=mex.set_crs("epsg:6372",allow_override=True)
 
mex=mex.to_crs("epsg:4326")   

fig, ax = plt.subplots(figsize=(16, 16))
mex.plot(ax=ax,color='snow', edgecolor='gray')
rectangles=df.geometry.rotate(21.018)
rectangles=rectangles.buffer(0.043,cap_style=3)
rectangles=rectangles.rotate(-21.018)

grid=gpd.GeoDataFrame(df,geometry=rectangles,crs="epsg:4326")

grid.plot(ax=ax,cmap="YlOrRd",alpha=0.6,legend=True,column=0,
          vmin=0,vmax=0.2,
        legend_kwds={'location':'bottom',
                     "shrink":0.5,
                     "label":"Slip Magnitude (m)",
                     "pad":0.04})
sfcoorx_tot= pd.read_table(FallaGuerrero.rootdir+FallaGuerrero.name+"/" + "sfcoorx_total.txt",sep="\s+",header=None,index_col=False )
sfcoory_tot= pd.read_table(FallaGuerrero.rootdir+FallaGuerrero.name+"/"+ "sfcoory_total.txt",sep="\s+",header=None,index_col=False )
sfcoorz_tot= pd.read_table(FallaGuerrero.rootdir+FallaGuerrero.name+"/"+ "sfcoorz_total.txt",sep="\s+",header=None,index_col=False )

sfcoorx_tot=sfcoorx_tot.to_numpy()
sfcoory_tot=sfcoory_tot.to_numpy()
sfcoorz_tot=abs(sfcoorz_tot.to_numpy()/(1e3))


from scipy.interpolate import griddata
from shapely.geometry import Polygon, MultiPolygon, LineString

def collec_to_gdf(collec_poly):
    """Transform a `matplotlib.contour.QuadContourSet` to a GeoDataFrame"""
    polygons = []
    for i, polygon in enumerate(collec_poly.collections):
        mpoly = []
        for path in polygon.get_paths():
            try:
                path.should_simplify = False
                poly = path.to_polygons()
                # Each polygon should contain an exterior ring + maybe hole(s):
                exterior, holes = [], []
                if len(poly) > 0 and len(poly[0]) > 3:
                    # The first of the list is the exterior ring :
                    exterior = poly[0]
                    # Other(s) are hole(s):
                    if len(poly) > 1:
                        holes = [h for h in poly[1:] if len(h) > 3]
                mpoly.append(Polygon(exterior, holes))
            except:
                print('Warning: Geometry error when making polygon #{}'
                      .format(i))
        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
    return gpd.GeoDataFrame(geometry=polygons, crs='epsg:6369')

xi = np.linspace(sfcoorx_tot.min(), sfcoorx_tot.max(), num=100)
yi = np.linspace(sfcoory_tot.min(), sfcoory_tot.max(), num=100)

#zi = griddata(x, y, z, xi, yi, interp='nn') # You could also take a look to scipy.interpolate.griddata
zi = griddata((sfcoorx_tot.flatten(), sfcoory_tot.flatten()), sfcoorz_tot.flatten(), (xi[None,:], yi[:,None]), method='cubic')

nb_class = [10,20,30,40,50,60,70,80,90,100] # Set the number of class for contour creation
# The class can also be defined by their limits like [0, 122, 333]
collec_poly=plt.contour(
    xi, yi, zi, nb_class, vmax=abs(zi).max(), vmin=-abs(zi).max(), colors="black")

gdf1 = collec_to_gdf(collec_poly)
geo=gdf1.to_json()
gdf1=gdf1.to_crs("epsg:4326")   
gdf1['geometry'] = gdf1.geometry.boundary
for cont,geo in enumerate(gdf1.geometry):
    gdf1['geometry'][cont]=LineString(geo.coords[:-1])
gdf1.plot(ax=ax,color="black",alpha=0.6)
xy=gdf1.representative_point()
xy=gpd.GeoDataFrame(geometry=xy,crs="epsg:4326")
xy.index=nb_class
xy.apply(lambda x: ax.annotate(text=x.name, xy=(x.geometry.x,x.geometry.y),
                               xytext=(0, 0), textcoords="offset points",
                               xycoords="data",
                               ha='center',fontstyle="italic",
                               family="serif",fontsize="medium"),axis=1)
#gdf.plot(ax=ax,color="greenyellow",marker="^",edgecolor="black",markersize=300)
plt.xlim([-102.5,-96])
plt.ylim([15.5,20])
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")

stats=Window1.stats_df(stats_name="selected_GPS_stats.dat")

gdf = gpd.GeoDataFrame(
    stats, geometry=gpd.points_from_xy(stats[1], stats[2]))
gdf.plot(ax=ax,color="greenyellow",marker="^",edgecolor="black",markersize=300)
dates=pd.read_table(Window1.wkdir+"windows_dates.in",sep="\s+",header=None,index_col=False)
plt.title("%s - %s with $\lambda_s$=%s and $\lambda_d$=%s"% (dates.at[0,0],dates.at[0,1],la1,la2),loc="center")
plt.show()
fig.savefig("Sismo_%s_%s.png" % (35,45), dpi=300)
#########################CV
fig, ax = plt.subplots(figsize=(16, 16))
mex.plot(ax=ax,color='snow', edgecolor='gray')
grid.plot(ax=ax,cmap="YlOrRd",alpha=0.6,legend=True,column="CV",
        legend_kwds={'location':'bottom',
                     "shrink":0.5,
                     "label":"CV(%)",
                     "pad":0.04})
gdf.plot(ax=ax,color="greenyellow",marker="^",edgecolor="black",markersize=300)
plt.xlim([-102.5,-96]) 
plt.ylim([15.5,20])
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.title("$\lambda_s$=%s and $\lambda_d$=%s"% (la1,la2),loc="center")
plt.show()
fig.savefig("CV_%s_%s.png" % (la1,la2), dpi=300)
# %%
Window2= Event.Event(FallaGuerrero, name="2019-2020 Window 2",
                           stats_file="selected_GPS_stats.dat", 
             U_file="data_windows_gps.in", NE_direction=True, Z_positive=False
             ,T_file="Okada_GPS_Guerrero_Oaxaca_2018_2020_10km.in")
AnalisisWindow2=Analysis.Analysis(Window1, Sn=0.0021, Se=0.0025, Sv=0.0051,
                               s_supp=(-0.1,0.1), d_supp=(-0.0804,0.4),beta1=1.,beta2=0.2 )

la1=35
la2=45
n=200000
BurnIn=6000
ss=5000
m=n+BurnIn
itera=1
AnalisisWindow2.Post(la1=la1, la2=la2, synt = False, la1_s = None, la2_s = None, sig2_betar = 0.0004)
simu,logEnergy = AnalisisWindow2.Run_MCMC(la1, la2, itera=itera , 
                    n = n, 
                    BurnIn = BurnIn, 
                    ss = ss,  
                    saveoutput = True, saveconfig=True)
CV=AnalisisWindow2.Compute_CV(la1=la1,la2=la2)
salida=Window2.wkdir+"Sim_la1_%s_la2_%s\salida/"%(la1,la2)
simu=np.loadtxt(salida+"sim%sm%s.txt"%(itera,m))
logEnergy=np.loadtxt(salida+"LogPos%sm%s.txt"%(itera,m))
CV=np.loadtxt(salida+"CV_%s_%s.txt"%(la1,la2))
fig1, axs = plt.subplots(1,)                        
plt.plot(logEnergy)
plt.title("logEnergy la1_%s_la2_%s" % (la1, la2))

di_tot=simu
di=np.median(di_tot,axis=0)
d=np.reshape(di,(len(di)//2,2))
slip=np.linalg.norm(d,axis=1)
slip=np.reshape(slip,FallaGuerrero.shape)

geometry=gpd.points_from_xy(sfcoorx.flatten(), sfcoory.flatten(),crs="epsg:6369")
df=gpd.GeoDataFrame(slip.flatten(),geometry=geometry,crs="epsg:6369")
df=df.assign(CV=CV)
df=df.to_crs("epsg:4326")

mex=gpd.read_file("C:/Users/migue/Downloads/889463770541_s/mg2022_integrado/conjunto_de_datos/00ent.shp")
mex=mex.set_crs("epsg:6372",allow_override=True)
 
mex=mex.to_crs("epsg:4326")   

fig, ax = plt.subplots(figsize=(16, 16))
mex.plot(ax=ax,color='snow', edgecolor='gray')

grid=gpd.GeoDataFrame(df,geometry=rectangles,crs="epsg:4326")

grid.plot(ax=ax,cmap="YlOrRd",alpha=0.6,legend=True,column=0,
          vmin=0,vmax=0.2,
        legend_kwds={'location':'bottom',
                     "shrink":0.5,
                     "label":"Slip Magnitude (m)",
                     "pad":0.04})

gdf1.plot(ax=ax,color="black",alpha=0.6)
xy.apply(lambda x: ax.annotate(text=x.name, xy=(x.geometry.x,x.geometry.y),
                               xytext=(0, 0), textcoords="offset points",
                               xycoords="data",
                               ha='center',fontstyle="italic",
                               family="serif",fontsize="medium"),axis=1)
#gdf.plot(ax=ax,color="greenyellow",marker="^",edgecolor="black",markersize=300)
plt.xlim([-102.5,-96])
plt.ylim([15.5,20])
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")

stats=Window2.stats_df(stats_name="selected_GPS_stats.dat")

gdf = gpd.GeoDataFrame(
    stats, geometry=gpd.points_from_xy(stats[1], stats[2]))
gdf.plot(ax=ax,color="greenyellow",marker="^",edgecolor="black",markersize=300)
dates=pd.read_table(Window2.wkdir+"windows_dates.in",sep="\s+",header=None,index_col=False)
plt.title("%s - %s with $\lambda_s$=%s and $\lambda_d$=%s"% (dates.at[0,0],dates.at[0,1],la1,la2),loc="center")
plt.show()
fig.savefig("Sismo_%s_%s.png" % (la1,la2), dpi=300)