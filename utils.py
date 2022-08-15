from __future__ import division
from math import sin, cos, acos, pi
import warnings
warnings.filterwarnings("ignore")
import numpy as np
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

def run_imap_multiprocessing(func, argument_list, show_prog = True):
    pool = mp.Pool(processes=mp.cpu_count())
    
    if show_prog:            
        result_list_tqdm = []
        for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list),position=0, leave=True):
            result_list_tqdm.append(result)
    else:
        result_list_tqdm = []
        for result in pool.imap(func=func, iterable=argument_list):
            result_list_tqdm.append(result)

    return result_list_tqdm

class curve_normalizer():
    def __init__(self, scale=True):
        """Intance of curve rotation and scale normalizer.
        Parameters
        ----------
        scale: boolean
                If true curves will be oriented and scaled to the range of [0,1]. Default: True.
        """
        self.scale = scale
        self.vfunc = np.vectorize(lambda c: self.get_oriented(c),signature='(n,m)->(n,m)')
        
    def get_oriented(self, curve):
        """Orient and scale(if enabled on initialization) the curve to the normalized configuration
        Parameters
        ----------
        curve: [n_point,2]
                Point coordinates describing the curve.

        Returns
        -------
        output curve: [n_point,2]
                Point coordinates of the curve oriented such that the maximum length is parallel to the x-axis and 
                scaled to have exactly a width of 1.0 on the x-axis is scale is enabled. Curve position is also 
                normalized to be at x=0 for the left most point and y=0 for the bottom most point.
        """
        ci = 0
        t = curve.shape[0]
        pi = t
        
        while pi != ci:
            pi = t
            t = ci
            ci = np.argmax(np.linalg.norm(curve-curve[ci],2,1))
        
        d = curve[pi] - curve[t]
        
        if d[1] == 0:
            theta = 0
        else:
            d = d * np.sign(d[1])
            theta = -np.arctan(d[1]/d[0])
        
        rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        out = np.matmul(rot,curve.T).T
        out = out - np.min(out,0)
        
        rot2 = np.array([[np.cos(theta+np.pi),-np.sin(theta+np.pi)],[np.sin(theta+np.pi),np.cos(theta+np.pi)]])
        out2 = np.matmul(rot2,curve.T).T
        out2 = out2 - np.min(out2,0)
        
        m1 = out[np.abs(out[:,0] - 0.5).argsort()[0:5],1].max()
        m2 = out2[np.abs(out2[:,0] - 0.5).argsort()[0:5],1].max()
        
        if m2<m1:
            out = out2
        
        if self.scale:
            out = out/np.max(out,0)[0]
        
        if np.isnan(np.sum(out)):
            out = np.zeros(out.shape)
                    
        return out
    
    def __call__(self, curves):
        """Orient and scale(if enabled on initialization) the batch of curve to the normalized configuration
        Parameters
        ----------
        curve: [N,n_point,2]
                batch of point coordinates describing the curves.

        Returns
        -------
        output curve: [N,n_point,2]
                Batch of point coordinates of the curve oriented such that the maximum length is parallel to the x-axis and 
                scaled to have exactly a width of 1.0 on the x-axis is scale is enabled. Curve position is also 
                normalized to be at x=0 for the left most point and y=0 for the bottom most point.
        """
        return self.vfunc(curves)