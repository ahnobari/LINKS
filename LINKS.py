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

from sim import *
from utils import *

pass