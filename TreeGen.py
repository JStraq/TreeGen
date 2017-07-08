# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 00:52:37 2017

@author: Joshua
"""

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as tkm
import numpy as np
#import matplotlib
#matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

from TreeLib import *
from TreeGUI import *

gui = TreeGUI()