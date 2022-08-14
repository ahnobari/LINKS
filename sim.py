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


def find_path(A, motor = [0,1], fixed_nodes=[0, 1]):
    
    # This function determines the path to the solution
    
    path = []
    
    A,fixed_nodes,motor = np.array(A),np.array(fixed_nodes),np.array(motor)
    
    unkowns = np.array(list(range(A.shape[0])))
    knowns = np.concatenate([fixed_nodes,[motor[-1]]])
    
    unkowns = unkowns[np.logical_not(np.isin(unkowns,knowns))]

    
    counter = 0
    while unkowns.shape[0] != 0:

        if counter == unkowns.shape[0]:
            # Non dyadic or DOF larger than 1
            return [], False
        n = unkowns[counter]
        ne = np.where(A[n])[0]
        
        kne = knowns[np.isin(knowns,ne)]
        
        if kne.shape[0] == 2:
            
            path.append([n,kne[0],kne[1]])
            counter = 0
            knowns = np.concatenate([knowns,[n]])
            unkowns = unkowns[unkowns!=n]
        elif kne.shape[0] > 2:
            #redundant or overconstraint
            return [], False
        else:
            counter += 1
    
    return np.array(path), True

def get_G(x0):
    return (np.linalg.norm(np.tile([x0],[x0.shape[0],1,1]) - np.tile(np.expand_dims(x0,1),[1,x0.shape[0],1]),axis=-1))

@tf.function
def get_GGpu(x0):
    return (tf.norm(tf.tile([x0],[x0.shape[0],1,1]) - tf.tile(tf.expand_dims(x0,1),[1,x0.shape[0],1]),axis=-1))


def solve(path,x0,G,motor,fixed_nodes,theta):
    # Basic Solver Algorithm
    
    path,x0,G,motor,fixed_nodes = np.array(path),np.array(x0),np.array(G),np.array(motor),np.array(fixed_nodes)
    
    x = np.zeros_like(x0)
    
    x[fixed_nodes] = x0[fixed_nodes]
    x[motor[1]] = x[motor[0]] + G[motor[0],motor[1]] * np.array([np.cos(theta),np.sin(theta)])
    
    for step in path:
        i = step[1]
        j = step[2]
        k = step[0]
        
        l_ij = np.linalg.norm(x[j]-x[i])
        cosphi = (l_ij **2 + G[i,k]**2 - G[j,k]**2)/(2 * l_ij * G[i,k])
        if cosphi >= -1.0 and cosphi <= 1.0:
            s = np.sign((x0[i,1]-x0[k,1])*(x0[i,0]-x0[j,0]) - (x0[i,1]-x0[j,1])*(x0[i,0]-x0[k,0]))
            phi = s * np.math.acos(cosphi)
            R = np.array([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), np.cos(phi)]])
            scaled_ij = (x[j]-x[i])/l_ij * G[i,k]
            x[k] = np.matmul(R, scaled_ij.reshape(2,1)).flatten() + x[i]
        else:
            #Locking or degenerate linkage!
            return x0, False, k
        
    return x, True, -1


def solve_full_revolution(path,A,x0,motor,fixed_nodes,n=200):
    # A function to solve for a full revolution of actuator Numerically
    
    path,x0,A,motor,fixed_nodes = np.array(path),np.array(x0),np.array(A),np.array(motor),np.array(fixed_nodes)
    
    thetas = np.linspace(0.0,2*np.pi,n)
    
    G = get_G(x0)
    
    solver = lambda t: solve(path,x0,G,motor,fixed_nodes,t)
    
    v_solve = np.vectorize(solver,signature='()->(n,2),(),()')
    
    return v_solve(thetas)

def solve_rev_vectorized(path,x0,G,motor,fixed_nodes,thetas):
    #Vectorized Faster Solver
    path,x0,G,motor,fixed_nodes = np.array(path),np.array(x0),np.array(G),np.array(motor),np.array(fixed_nodes)
    
    x = np.zeros([x0.shape[0],thetas.shape[0],2])
    
    x[fixed_nodes] = np.expand_dims(x0[fixed_nodes],1)
    x[motor[1]] = x[motor[0]] + G[motor[0],motor[1]] * np.concatenate([[np.cos(thetas)],[np.sin(thetas)]]).T
    
    state = np.zeros(thetas.shape[0])
    flag = True
    kk = np.zeros(thetas.shape[0]) - 1.0
    
    for step in path:
        i = step[1]
        j = step[2]
        k = step[0]
        
        l_ij = np.linalg.norm(x[j]-x[i],axis=-1)
        cosphi = (l_ij ** 2 + G[i,k]**2 - G[j,k]**2)/(2 * l_ij * G[i,k])

        state += np.logical_or(cosphi<-1.0,cosphi>1.0)
        
        kk = state * k * (kk==-1.0) + kk
        
        s = np.sign((x0[i,1]-x0[k,1])*(x0[i,0]-x0[j,0]) - (x0[i,1]-x0[j,1])*(x0[i,0]-x0[k,0]))

        phi = s * np.arccos(cosphi)

        a = np.concatenate([[np.cos(phi)],[-np.sin(phi)]]).T
        b = np.concatenate([[np.sin(phi)],[np.cos(phi)]]).T

        R = np.swapaxes(np.concatenate([[a],[b]]),0,1)

        scaled_ij = (x[j]-x[i])/np.expand_dims(l_ij,-1) * G[i,k]
        x[k] = np.squeeze(R @ np.expand_dims(scaled_ij,-1)) + x[i]
        
    kk = (kk!=-1.0) + kk
    
    return x, state == 0.0, kk.astype(np.int32)


@tf.function
def solve_rev_vectorized_GPU(path,x0,G,motor,fixed_nodes,thetas):
    #GPU Implementation
    x = tf.zeros([x0.shape[0],thetas.shape[0],2])
    
    x = x + tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.one_hot(fixed_nodes,x0.shape[0]),0),-1) * x0,1)
    m = x[motor[0]] + G[motor[0],motor[1]] * tf.transpose(tf.concat([[tf.cos(thetas)],[tf.sin(thetas)]],0))
    m = tf.expand_dims(m,0)
    m = tf.pad(m,[[motor[1],x0.shape[0]-motor[1]-1],[0,0],[0,0]])
    x = x + m
    
    state = tf.zeros(thetas.shape[0])
    flag = True
    
    cosphis = tf.zeros([path.shape[0],thetas.shape[0]])
    
    for st in range(path.shape[0]):
        i = path[st][1]
        j = path[st][2]
        k = path[st][0]
        
        l_ij = tf.norm(x[j]-x[i],axis=-1)
        cosphi = (l_ij ** 2 + G[i,k]**2 - G[j,k]**2)/(2 * l_ij * G[i,k])
        cosphis = cosphis + tf.pad(tf.expand_dims(cosphi,0),[[st,path.shape[0]-st-1],[0,0]])
        
        state += tf.cast(tf.logical_or(cosphi<-1.0,cosphi>1.0),tf.float32)
        
        s = tf.sign((x0[i,1]-x0[k,1])*(x0[i,0]-x0[j,0]) - (x0[i,1]-x0[j,1])*(x0[i,0]-x0[k,0]))

        phi = s * tf.acos(cosphi)

        a = tf.transpose(tf.concat([[tf.cos(phi)],[-tf.sin(phi)]],0))
        b = tf.transpose(tf.concat([[tf.sin(phi)],[tf.cos(phi)]],0))

        R = tf.einsum("ij...->ji...", tf.concat([[a],[b]],0))
        
        scaled_ij = (x[j]-x[i])/tf.expand_dims(l_ij,-1) * G[i,k]
        x_k = tf.squeeze(tf.matmul(R, tf.expand_dims(scaled_ij,-1))) + x[i]
        x_k = tf.expand_dims(x_k,0)
        x_k = tf.pad(x_k,[[k,x0.shape[0]-k-1],[0,0],[0,0]])
        x = x + x_k
          
    return x, state == 0.0, cosphis

@tf.function
def solve_rev_vectorized_batch_GPU(As,x0s,Gs,node_types,thetas):
    # Batch Fully Vectorized Solver
    # Note All Mecanisms in the batch are assumed to be sorted in order of path this will fail if this condition is not met
    
    x = tf.zeros([tf.shape(x0s)[0],tf.shape(x0s)[1],thetas.shape[0],2])
    
    x = x + tf.expand_dims(node_types * x0s,2)
    
    m = x[:,0] + tf.tile(tf.expand_dims(tf.transpose(tf.concat([[tf.cos(thetas)],[tf.sin(thetas)]],0)),0),[tf.shape(x0s)[0],1,1]) * tf.expand_dims(tf.expand_dims(Gs[:,0,1],-1),-1)
    m = tf.expand_dims(m,1)
    m = tf.pad(m,[[0,0],[1,tf.shape(x0s)[1]-2],[0,0],[0,0]])
    x = x + m
    
    ds = tf.zeros([tf.shape(x0s)[0],tf.shape(x0s)[1],thetas.shape[0]])
    state = tf.zeros([tf.shape(x0s)[0],thetas.shape[0]])
    
    for k in range(3,tf.shape(x0s)[1]):
        
        inds = tf.argsort(As[:,k,0:k])[:,-2:]
        
        l_ijs = tf.norm(tf.gather_nd(x,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,0]],-1)) - tf.gather_nd(x,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,1]],-1)), axis=-1)
        
        gik = tf.expand_dims(tf.gather_nd(Gs,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,0],tf.ones([tf.shape(x0s)[0]],dtype=tf.int32)*k],-1)),-1)
        gjk = tf.expand_dims(tf.gather_nd(Gs,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,1],tf.ones([tf.shape(x0s)[0]],dtype=tf.int32)*k],-1)),-1)
        
        cosphis = (tf.square(l_ijs) + tf.square(gik) - tf.square(gjk))/(2 * l_ijs * gik)
        
        cosphis = tf.where(tf.tile(node_types[:,k],[1,thetas.shape[0]])==0.0,cosphis,tf.zeros_like(cosphis))
        
        state += tf.cast(tf.logical_or(cosphis<-1.0,cosphis>1.0),tf.float32)
        
        dts = 1 - tf.square(cosphis)
        dts = tf.expand_dims(dts,1)
        dts = tf.pad(dts,[[0,0],[k,tf.shape(x0s)[1]-k-1],[0,0]])
        ds = ds + dts
        
        x0i1 = tf.gather_nd(x0s,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,0],tf.ones(tf.shape(x0s)[0],dtype=tf.int32)],-1))
        x0i0 = tf.gather_nd(x0s,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,0],tf.zeros(tf.shape(x0s)[0],dtype=tf.int32)],-1))
        
        x0j1 = tf.gather_nd(x0s,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,1],tf.ones(tf.shape(x0s)[0],dtype=tf.int32)],-1))
        x0j0 = tf.gather_nd(x0s,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,1],tf.zeros(tf.shape(x0s)[0],dtype=tf.int32)],-1))
        
        x0k1 = x0s[:,k,1]
        x0k0 = x0s[:,k,0]
        
        s = tf.expand_dims(tf.sign((x0i1-x0k1)*(x0i0-x0j0) - (x0i1-x0j1)*(x0i0-x0k0)),-1)
        

        phi = s * tf.acos(cosphis)
        
        a = tf.transpose(tf.concat([[tf.cos(phi)],[-tf.sin(phi)]],0),perm=[1,2,0])
        b = tf.transpose(tf.concat([[tf.sin(phi)],[tf.cos(phi)]],0),perm=[1,2,0])

        R = tf.einsum("ijk...->jki...", tf.concat([[a],[b]],0))
        
        xi = tf.gather_nd(x,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,0]],-1))
        xj = tf.gather_nd(x,tf.stack([tf.range(tf.shape(x0s)[0]),inds[:,1]],-1))
        
        scaled_ij = (xj-xi)/tf.expand_dims(l_ijs,-1) * tf.expand_dims(gik,-1)
        
        x_k = tf.squeeze(tf.matmul(R, tf.expand_dims(scaled_ij,-1))) + xi
        x_k = tf.where(tf.tile(tf.expand_dims(node_types[:,k],-1),[1,thetas.shape[0],2])==0.0,x_k,tf.zeros_like(x_k))
        x_k = tf.expand_dims(x_k,1)
        x_k = tf.pad(x_k,[[0,0],[k,tf.shape(x0s)[1]-k-1],[0,0],[0,0]])
        x = x + x_k
          
    return x,ds,state == 0.0

def get_re_nodes(A,target):
    #This function reduced a mechanism to a given joint (Output is a list of joint indecies that are needed)

    neis = np.where(A[target])[0]
    neis = neis[neis<target]
    nodes = list(neis)
    if neis.shape[0] == 1:
        nodes += get_re_nodes(A,neis[0])
    elif neis.shape[0] == 2:
        nodes += get_re_nodes(A,neis[0]) + get_re_nodes(A,neis[1])
    
    return list(set(nodes + [target]))

def simulate_mechanism(M,n=200):
    #This function produces the data needed for the dataset
    A, x0, node_type = M
    
    fixed_nodes = np.where(node_type)[0]
    
    moving_nodes = np.where(node_type==0)[0]
    
    path = find_path(A,fixed_nodes=fixed_nodes)[0]
    
    x_sol = solve_rev_vectorized(path,x0,get_G(x0),[0,1],fixed_nodes,np.linspace(0,2*np.pi,n))[0]
    
    if np.isnan(x_sol).sum() != 0:
        return None
    
    x_sol_n = normalizer_object(x_sol)[moving_nodes]
    
    x_norm_cur = []
    x_norm_cur_i = []
    
    for i,mn in enumerate(moving_nodes):
        if np.isin(np.where(A[mn]),fixed_nodes).sum() != 0:
            if np.random.uniform()>=0.995:
                x_norm_cur.append(x_sol_n[i])
                x_norm_cur_i.append(mn)
        
        elif np.linalg.norm(x_sol_n[i] - [0.5,0.5],axis=-1).var()< 5e-4:
            if np.random.uniform()>=0.995:
                x_norm_cur.append(x_sol_n[i])
                x_norm_cur_i.append(mn)
        
        else:
            x_norm_cur.append(x_sol_n[i])
            x_norm_cur_i.append(mn)
            
    
    return x_sol, x_sol_n, moving_nodes, np.array(x_norm_cur), np.array(x_norm_cur_i)

def draw_mechanism(A,x0,fixed_nodes,motor, highlight=100, solve=True, thetas = np.linspace(0,np.pi*2,200), def_alpha = 1.0, h_alfa =1.0, h_c = "#f15a24"):
    
    #Draws a given mechanism

    def fetch_path():
        root = etree.parse(StringIO('<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 338"><defs><style>.cls-1{fill:#1a1a1a;stroke:#1a1a1a;stroke-linecap:round;stroke-miterlimit:10;stroke-width:20px;}</style></defs><path class="cls-1" d="M45.5,358.5l70.71-70.71M46,287.5H644m-507.61,71,70.72-70.71M223,358.5l70.71-70.71m20.18,70.72,70.71-70.71m13.67,70.7,70.71-70.71m20.19,70.72,70.71-70.71m15.84,70.71,70.71-70.71M345,39.62A121.38,121.38,0,1,1,223.62,161,121.38,121.38,0,0,1,345,39.62Z" transform="translate(-35.5 -29.62)"/></svg>')).getroot()
        view_box = root.attrib.get('viewBox')
        if view_box is not None:
            view_box = [int(x) for x in view_box.split()]
            xlim = (view_box[0], view_box[0] + view_box[2])
            ylim = (view_box[1] + view_box[3], view_box[1])
        else:
            xlim = (0, 500)
            ylim = (500, 0)
        path_elem = root.findall('.//{http://www.w3.org/2000/svg}path')[0]
        return xlim, ylim, parse_path(path_elem.attrib['d'])
    _,_,p = fetch_path()
    p.vertices -= p.vertices.mean(axis=0)
    p.vertices = (np.array([[np.cos(np.pi), -np.sin(np.pi)],[np.sin(np.pi), np.cos(np.pi)]])@p.vertices.T).T
    
    A,x0,fixed_nodes,motor = np.array(A),np.array(x0),np.array(fixed_nodes),np.array(motor)
    
    x = x0
    
    N = A.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            if i == highlight:
                plt.scatter(x[i,0],x[i,1],color=h_c,s=700,zorder=10,marker=p)
            else:
                plt.scatter(x[i,0],x[i,1],color="#1a1a1a",s=700,zorder=10,marker=p)
        else:
            if i == highlight:
                plt.scatter(x[i,0],x[i,1],color=h_c,s=100,zorder=10,facecolors=h_c,alpha=0.7)
            else:
                plt.scatter(x[i,0],x[i,1],color="#1a1a1a",s=100,zorder=10,facecolors='#ffffff',alpha=0.7)
        
        for j in range(i+1,N):
            if A[i,j]:
                if (motor[0] == i and motor[1] == j) or(motor[0] == j and motor[1] == i):
                    plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#ffc800",linewidth=4.5)
                else:
                    plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#1a1a1a",linewidth=4.5,alpha=0.6)
                
    if solve:
        path = find_path(A,motor,fixed_nodes)[0]
        G = get_G(x0)
        x,c,k =  solve_rev_vectorized(path.astype(np.int32), x0, G, motor, fixed_nodes,thetas)
        x = np.swapaxes(x,0,1)
        if np.sum(c) == c.shape[0]:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        plt.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
        else:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        plt.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
            plt.text(0.5, 0.5, 'Locking Or Under Defined', color='red', horizontalalignment='center', verticalalignment='center')
        
    plt.axis('equal')
    plt.axis('off')