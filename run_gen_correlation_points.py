
import pickle

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os


# =================================
# verify and save the data

def dist(coords1,coords2):
    return ((coords1[0]-coords2[0])**2+(coords1[1]-coords2[1])**2)**0.5

def validate_and_save(filename,coordsss,lattice_size,override=False):
    if not override and os.path.exists(filename):
        print('file already exists:',filename)
        return
    # remove the duplicated ones
    coordsss=list(sorted(set(coordsss)))
    print('filename:',filename)
    for coordss in coordsss:
        # print coords and distance
        for coords in coordss:
            print(coords,end=' ')
        print()
        for i in range(len(coordss)):
            for j in range(i+1,len(coordss)):
                print(dist(coordss[i],coordss[j]),end=' ')
        print()
        # check if they are the same point
        for i in range(len(coordss)):
            for j in range(i+1,len(coordss)):
                assert coordss[i]!=coordss[j]
        # check if they are in the range of the lattice
        for coords in coordss:
            assert coords[0]>=0 and coords[0]<lattice_size[0]
            assert coords[1]>=0 and coords[1]<lattice_size[1]

    pickle.dump(coordsss,open(filename,'wb'))
    print('total correlators:',len(coordsss))
    print('saved to',filename)


def generate_2pt_correlation_points(lattice_size,data_count=100,fixed_x0=None,fixed_y0=None):
    rmin,rmax=1,min(lattice_size)
    coordsss=[]
    while len(coordsss)<data_count:
        th=np.random.uniform(0,np.pi/2)
        r=np.exp(np.random.uniform(np.log(rmin),np.log(rmax)))
        x,y=int(np.abs(r*np.cos(th))),int(np.abs(r*np.sin(th)))
        if x==0 and y==0:
            x,y=(1,0) if np.random.uniform()<0.5 else (0,1)
        x0,y0=np.random.randint(0,lattice_size[0]-x),np.random.randint(0,lattice_size[1]-y)
        x0=fixed_x0 if fixed_x0 is not None else x0
        y0=fixed_y0 if fixed_y0 is not None else y0
        x1,y1=x0+x,y0+y
        coordsss.append(((x0,y0),(x1,y1)))
    return coordsss

lattice_size=(2**30,2**30)
coordsss=generate_2pt_correlation_points(lattice_size,data_count=100)
validate_and_save('data/2pt_correlation_points_30.pkl',coordsss,lattice_size)
coordsss=generate_2pt_correlation_points(lattice_size,data_count=900)
validate_and_save('data/2pt_correlation_points_30_appended.pkl',coordsss,lattice_size)
