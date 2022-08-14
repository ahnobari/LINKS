from __future__ import division
from math import sin, cos, acos, pi
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pymoo
from pymoo.util.display import Display
from tqdm.autonotebook import trange
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import csv
from pymoo.factory import get_performance_indicator
import os
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib.patches import PathPatch

from io import StringIO
import requests
import xml.etree.ElementTree as etree

from svgpath2mpl import parse_path

from sim import *
from utils import *

def get_random_topology(n=None,N_min=8,N_max=20,g_pob=0.25):
    # This function generates random skeletons with 1 DOF and no redundancy

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    
    if n is None:
        n = np.random.randint(low=N_min,high=N_max+1)
        
    fixed_nodes = [0,2]
    simples_nodes = [1]
    motor = [0,1]
    A = np.zeros([n,n]).astype(np.int8)
    A[0,1] = 1
    A[1,0] = 1
    
    for i in range(3,n):
        nodes_lack = np.where(np.triu(A).sum(-1)[:i]==0)[0]
        if A[fixed_nodes[-1]].sum() != 0 and np.random.uniform()<=g_pob and i<n-2 and nodes_lack.shape[0] < n-i:
            fixed_nodes.append(i)
        elif i == n-1:
            if nodes_lack.shape[0] > 1:
                p = np.random.choice(nodes_lack,size=2,replace=False)
                A[i,p[0]] = 1
                A[p[0],i] = 1
                A[i,p[1]] = 1
                A[p[1],i] = 1
            else:
                p = np.random.choice(simples_nodes[0:-1],size=1,replace=False)
                A[i,p] = 1
                A[p,i] = 1
                A[i-1,i] = 1
                A[i,i-1] = 1
                simples_nodes.append(i)
        elif i == n-2:
            
            if A[fixed_nodes[-1]].sum() == 0:
                A[i,fixed_nodes[-1]] = 1
                A[fixed_nodes[-1],i] = 1
                if nodes_lack.shape[0] >= n-i:
                    nodes_lack = nodes_lack.tolist()
                    nodes_lack.pop(nodes_lack.index(fixed_nodes[-1]))
                    p = np.random.choice(nodes_lack)
                    A[i,p] = 1
                    A[p,i] = 1
            else:
                if nodes_lack.shape[0] >= n-i:
                    p = np.random.choice(nodes_lack,size = 2, replace = False)
                    A[i,p[0]] = 1
                    A[p[0],i] = 1
                    A[i,p[1]] = 1
                    A[p[1],i] = 1
                else:
                    p = np.random.choice(simples_nodes)
                    A[i,p] = 1
                    A[p,i] = 1
                    simples_nodes.pop(simples_nodes.index(p))
                    pp = np.random.choice(simples_nodes+fixed_nodes)
                    A[i,pp] = 1
                    A[pp,i] = 1
                    simples_nodes.append(p)
            simples_nodes.append(i)
        else:
            if nodes_lack.shape[0] >= n-i:
                p = np.random.choice(nodes_lack,size = 2, replace = False)
                A[i,p[0]] = 1
                A[p[0],i] = 1
                A[i,p[1]] = 1
                A[p[1],i] = 1
            else:
                p = np.random.choice(simples_nodes)
                A[i,p] = 1
                A[p,i] = 1
                simples_nodes.pop(simples_nodes.index(p))
                pp = np.random.choice(simples_nodes+fixed_nodes)
                A[i,pp] = 1
                A[pp,i] = 1
                simples_nodes.append(p)
            simples_nodes.append(i)
    
    return A,motor,fixed_nodes

def get_configs_for_topology(path,A,motor,fixed_nodes,max_iter=1000):
    #Finds Initial Positions For a Given Topology

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    
    k = 0
    x0 = np.zeros([A.shape[0],2])
    for i in range(max_iter):

        x0_t = np.random.uniform(size=[A.shape[0],2])
        x0_t[0] = [0.5,0.5]
        x0_t[1] = [0.55,0.5]
        x0[k:] = x0_t[k:]
        
        G = get_G(x0)
        
        x,c,k =  solve_rev_vectorized(path,x0,G,motor,fixed_nodes,np.linspace(0,np.pi*2,10))

        if c.sum() == c.shape[0]:
            x,c,k =  solve_rev_vectorized(path,x0,G,motor,fixed_nodes,np.linspace(0,np.pi*2,200))
            if c.sum() == c.shape[0]:
                return x0
            else:
                k = k * 0
        k = k[k>=0].min()
        
    return get_configs_for_topology(path,A,motor,fixed_nodes,max_iter)


def get_candidates_for_topology(top,n=5):
    #run the previous function a certain number of times for a given topology
    A,motor,fixed_nodes = top
    path = find_path(A,motor,fixed_nodes)[0]
    x0s = []
    max_iter = n*5
    co = 0
    while len(x0s)<n:
        c = get_configs_for_topology(path,A,motor,fixed_nodes)
        if np.sum(c<0) == 0:
            x0s.append(c)
        co+=1
        
    return x0s