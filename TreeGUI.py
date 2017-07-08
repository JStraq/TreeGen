# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:49:21 2017

@author: Joshua
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 00:52:37 2017

@author: Joshua
"""

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as tkm
import numpy as np
import pickle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

from TreeLib import *

class TreeGUI:
    def __init__(self):
        
        self.history = []
        self.grid = [[None]]
        self.chapter = None
        
        self.window = tk.Tk()
        self.window.wm_title("KKPsi Family Tree Generator v1.0")
        self.fulldraw = tk.BooleanVar(self.window)
        self.fulldraw.set(False)
        self.xaspect = tk.DoubleVar(self.window)
        self.xaspect.set(16)
        self.yaspect = tk.DoubleVar(self.window)
        self.yaspect.set(9)
        self.linewidth = tk.DoubleVar(self.window)
        self.linewidth.set(2)

        self.frameCtrl = tk.Frame(self.window, height=500, width = 50)
        self.frameCtrl.grid(row=0, column=0, sticky='NSEW', padx=10, pady=10)
        self.frameCtrl.grid_columnconfigure(0, weight=1)
        
        self.bLoadData = tk.Button(self.frameCtrl, text='Load Raw Data', command=self.loadFile)
        self.bLoadData.grid(row=0, column=0, sticky='NSEW')
        self.bLoadConfig = tk.Button(self.frameCtrl, text='Load Tree File', command=self.loadConfig)
        self.bLoadConfig.grid(row=1, column=0, sticky='NSEW')
        self.bSaveConfig = tk.Button(self.frameCtrl, text='Save Tree File', command=self.saveConfig)
        self.bSaveConfig.grid(row=3, column=0, sticky='NSEW')
        self.bSavePic = tk.Button(self.frameCtrl, text='Save Image', command=self.saveImage)
        self.bSavePic.grid(row=4, column=0, sticky='NSEW')
        
        self.bTight = tk.Button(self.frameCtrl, text='Tighten', command=self.tightenGrid)
        self.bTight.grid(row=5, column=0, sticky='NSEW')
        self.bCompact = tk.Button(self.frameCtrl, text='Compactify', command=self.compactifyGrid)
        self.bCompact.grid(row=6, column=0, sticky='NSEW')
        self.bReorder = tk.Button(self.frameCtrl, text='Reorder', command=self.reorderFams)
        self.bReorder.grid(row=7, column=0, sticky='NSEW')
        
        self.bEdit = tk.Button(self.frameCtrl, text='Edit Plot', command=self.editPlot)
        self.bEdit.grid(row=8, column=0, sticky='NSEW')
        self.bEdit = tk.Button(self.frameCtrl, text='Redraw', command=self.updatePlot)
        self.bEdit.grid(row=9, column=0, sticky='NSEW')
        
        
        self.figure = Figure(figsize=(12, 8), dpi=50)
        self.framePlt = tk.Frame(self.window)#, height=500, width=600)
        self.framePlt.grid(row=0, column=1, sticky='NSEW')
        self.framePlt.grid_columnconfigure(0, weight=1)
        self.framePlt.grid_rowconfigure(0, weight=1)
        self.framePlt.grid_rowconfigure(1, weight=0)        
        
        self.plotCanvas = FigureCanvasTkAgg(self.figure, self.framePlt)
        self.plotCanvas.show()
        self.plotCanvas.get_tk_widget().grid(row=0, column=0, sticky='NSEW', columnspan=2)
        
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)
        
        tk.mainloop()

    def loadFile(self):
        self.plotFileName = tk.filedialog.askopenfilename(title = "Select a raw data file",
                                                          filetypes = (("Data Files","*.csv"),("all files","*.*")))
        self.chapter = Chapter(self.plotFileName)
        self.grid = initGrid(self.chapter,n=10)
        self.grid = orgTree(self.grid)
        self.updatePlot()
        
    def tightenGrid(self):
        self.grid, loners = stashLoners(self.grid)
        self.grid = tighten(self.grid)
        self.grid = replaceLoners(self.grid, loners)
        self.updatePlot()
        
    def compactifyGrid(self):
        self.grid, loners = stashLoners(self.grid)
        self.grid = compactify(self.grid)
        self.grid = replaceLoners(self.grid, loners)
        self.updatePlot()
    
    def reorderFams(self):
        self.grid, loners = stashLoners(self.grid)
        self.grid = jigsaw(self.grid)
        self.grid = replaceLoners(self.grid, loners)
        self.updatePlot()
        
    def updatePlot(self, *args):                                      
        if self.fulldraw.get():
            drawTree(self.grid, self.chapter, self.figure, aspect=self.yaspect.get()/self.xaspect.get(), linewidth=self.linewidth.get())
        else:
            drawTreeFast(self.grid, self.figure, aspect=self.yaspect.get()/self.xaspect.get(), linewidth=self.linewidth.get())
        
        self.figure.tight_layout()
        
        self.plotCanvas.show()
    
    def saveConfig(self):
        savepath = self.imageFileName = tk.filedialog.asksaveasfilename(title = "Choose a config name",
                                                                        filetypes = (("Tree Files","*.pickle"),("all files","*.*")))
        if savepath[:-7] != '.pickle':
            savepath += '.pickle'
        try:
            with open(savepath, 'wb') as f:
                pickle.dump(self.grid, f, pickle.HIGHEST_PROTOCOL)
        except:
            pass

    def loadConfig(self):
        loadpath = self.imageFileName = tk.filedialog.askopenfilename(title = "Select a config file",
                                                                      filetypes = (("Tree Files","*.pickle"),("all files","*.*")))
        try:
            with open(loadpath, 'rb') as f:
                self.grid = pickle.load(f)
            
            self.chapter = Chapter()
            for row in self.grid:
                for bro in row:
                    if bro is not None:
                        self.chapter.appendBro(bro)
        except FileNotFoundError:
            pass
        self.updatePlot()
    
    def saveImage(self):
        top = tk.Toplevel(self.window)
        tk.Label(top, text='Resolution (dpi) = ').grid(row=0, column=0, sticky='NSE')
        self.resVar = tk.IntVar()
        self.resVar.set('100')
        self.xsize = tk.IntVar()
        self.ysize = tk.IntVar()
        self.calcSize()
        resBox = tk.Entry(top, width=20, textvariable=self.resVar)
        resBox.grid(row=0, column=1, columnspan=2, sticky='NSW')
        resBox.bind('<FocusOut>', self.calcSize)
        resBox.bind('<Return>', self.calcSize)
        
        tk.Label(top, textvariable=self.xsize).grid(row=1, column=0, sticky='NSE')
        tk.Label(top, text=' x ').grid(row=1, column=1, sticky='NSEW')
        tk.Label(top, textvariable=self.ysize).grid(row=1, column=2, sticky='NSW')
        
        tk.Button(top, text='Save PNG', command=lambda: self.makeImage(top)).grid(row=2,column=0, columnspan=3, sticky='NSEW')
    
    def calcSize(self, *args):
        self.xsize.set(self.resVar.get()*self.figure.get_size_inches()[0])
        self.ysize.set(self.resVar.get()*self.figure.get_size_inches()[1])
        
    def makeImage(self, top):
        self.imageFileName = tk.filedialog.asksaveasfilename()
        self.figure.savefig(self.imageFileName, dpi=int(self.resVar.get()), bbox_inches="tight")
        top.destroy()
        
    def editPlot(self):
        top = tk.Toplevel()
        tk.Checkbutton(top, text="Show names", variable=self.fulldraw, onvalue=True, offvalue=False).grid(row=0, column=0, sticky='NSEW')
        
        tk.Label(top, text='Linewidth = ').grid(row=1,column=0, sticky='NSEW')
        tk.Entry(top, width=10, textvariable=self.linewidth).grid(row=1, column=1, sticky='NSEW')
        
        tk.Label(top, text='Aspect Ratio').grid(row=2,column=0, sticky='NSEW')
        tk.Entry(top, width=5, textvariable=self.xaspect).grid(row=2, column=1, sticky='NSEW')
        tk.Label(top, text=':').grid(row=2,column=2, sticky='NSEW')
        tk.Entry(top, width=5, textvariable=self.yaspect).grid(row=2, column=3, sticky='NSEW')
        top.protocol("WM_DELETE_WINDOW", lambda: self.updateConfigs(top))
    
    def updateConfigs(self, top):
        self.updatePlot()
        top.destroy()