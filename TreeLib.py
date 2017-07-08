# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 08:34:18 2017

@author: Joshua
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import groupby

alphabet = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
                    'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho',
                    'Sigma', 'Tau', 'Upsilon',  'Phi', 'Chi', 'Psi', 'Omega']

def classnum(cls):
        num = 0
        letters = cls.split(' ')
        for ii,letter in enumerate(letters[::-1]):
            if letter=='Founders':
                num=-1
            else:
                num += (alphabet.index(letter)+1) * len(alphabet)**ii
                if ii==0:
                    num -= 1
        return num

class Chapter:
    def __init__(self, path=None):
        self.bros = []
        if path is not None:
            self.readData(path)
        self.classes = self.getClasses()
        
    def addBro(self, name, pmclass, big=None):  # all strings
        if big is not None:
            for b in self.bros:
                if big==b.name:
                    big = b
                    break
        self.bros.append(Bro(name, pmclass, big))
    
    def appendBro(self, bro):
        self.bros.append(bro)
    
    def getClasses(self):
        classes = []
        for bro in self.bros:
            if bro.pmclass not in classes:
                classes.append(bro.pmclass)
        
        clsorder = []
        for cls in classes:
            clsorder.append(classnum(cls))
        return [classes[ii] for ii in np.argsort(clsorder)]
    
    
    def readData(self, path):
        file = open(path, 'r')
        lines = file.readlines()
        lines = [line.strip('\n') for line in lines]
        for ii in range(int(len(lines)/3)+1):
            cls = lines[3*ii].split(',')[0]
            names = lines[3*ii].split(',')[1:]
            bigs = lines[3*ii+1].split(',')[1:]
            for jj, name in enumerate(names):
                if name.strip() != '':
                    big = bigs[jj].strip()
                    if big=='':
                        big=None
                    else:
                        for bro in self.bros:
                            if bro.name == big:
                                big = bro
                        if isinstance(big, str):
                            big = None

                    self.addBro(name, cls, big)
        for bro in self.bros:
            bro.count = bro.calcCount()
    
    def minwidth(self):
        width = 0
        for cls in self.classes:
            classsize = 0
            for bro in self.bros:
                if bro.pmclass == cls:
                    classsize += 1
            width = max((width, classsize))
        return width
        
class Bro:
    def __init__(self, name, pmclass, big=None):
        self.name = name
        self.pmclass= pmclass
        self.littles = []
        self.big = big
        if big is not None:
            self.big.addLittle(self)
        self.count = 0
    
    def addLittle(self, little):
        self.littles.append(little)
    
    def calcCount(self):
        count = 0
        if len(self.littles)==0:
            return 0
        else:
            for little in self.littles:
                count = count + 1
                count = count + little.calcCount()
        return count
    
    def isSib(self, bro):
        return (self.big==bro.big) and (self != bro)
    
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

def initGrid(chapter, n=10):
    width = chapter.minwidth()*n
    length = len(chapter.classes)
    grid = [[None for x in range(width)] for y in range(length)]
    for ii in range(length):
        jj = 0
        for bro in chapter.bros:
            if bro.pmclass==chapter.classes[ii]:
                grid[ii][jj] = bro
                jj+=1
    for ii in range(len(grid)):
        np.random.shuffle(grid[ii])
    return grid
    
def expandGrid(grid, n=5):
    length = len(grid)
    width = len(grid[0])
    for row in range(length):
        for col in range(width)[::-1]:
            for x in range(n):
                grid[row].insert(col, None)
        for x in range(n):
            grid[row].append(None)
    return grid

def collapseGrid(grid):
    length = len(grid)
    width = len(grid[0])
    for col in range(width)[::-1]:
        colcheck = [grid[row][col] is None for row in range(length)]
        if False not in colcheck:
            for row in range(length):
                del grid[row][col]
    return grid

def drawTree(grid, chapter, figure, aspect, linewidth):
    figure.clear()
    subplot = figure.add_subplot(111)
    
    figwidth, figheight = figure.get_size_inches() * figure.dpi
    length = len(grid)
    width = len(grid[0])
    
    height = figheight if (figheight/figwidth < aspect) else (figwidth*aspect)
    font = height/(3.5*length)
    linewidth = linewidth*font/10
    
    subplot.set_xlim(-3,width)
    subplot.set_ylim(-1,length)
    subplot.axis('off')
    subplot.set_aspect(aspect*(width+3)/(length+1))

    for row, cls in enumerate(chapter.classes):  # print class names
        subplot.text(-2, length-row-1, cls.upper(), ha='center', va='center', fontweight = 'bold', fontsize= font)
    
    for row in range(length):
        for col in range(width):
            if grid[row][col] is not None:
                if '(' in str(grid[row][col]):
                    formatted = str(grid[row][col]).replace(' ','\n',2)
                elif '-' in str(grid[row][col]):
                    formatted = str(grid[row][col]).replace('-','-\n',1)
                else:
                    formatted = str(grid[row][col]).replace(' ','\n',1)
                subplot.text(col, length-row-1, formatted, ha='center', va='center', fontsize= font)

                if grid[row][col].big is not None:
                    for brow in range(row):
                        for bcol in range(width):
                            if grid[brow][bcol] == grid[row][col].big:
                                if col==bcol:
                                    subplot.plot([col, bcol], [length-row-0.7, length-brow-1.3], color='k',
                                             linestyle='-', linewidth=linewidth)
                                    pass
                                else:
                                    subplot.plot([bcol, bcol], [length-brow-1.3, length-brow-1.5], color='k',
                                             linestyle='-', linewidth=linewidth)
                                    subplot.plot([bcol, col], [length-brow-1.5, length-brow-1.5], color='k',
                                             linestyle='-', linewidth=linewidth)
                                    subplot.plot([col, col], [length-row-0.7, length-brow-1.5], color='k',
                                             linestyle='-', linewidth=linewidth)


def drawTreeFast(grid, figure, aspect, linewidth):
    figure.clear()
    subplot = figure.add_subplot(111)
    length = len(grid)
    width = len(grid[0])

    subplot.set_xlim(-1,width)
    subplot.set_ylim(-1,length)
    subplot.axis('off')
    
    figwidth, figheight = figure.get_size_inches() * figure.dpi
    length = len(grid)
    width = len(grid[0])
    
    height = figheight if (figheight/figwidth < aspect) else (figwidth*aspect)
    linewidth = linewidth*height/(35*length)
    size = height/20
        
    subplot.set_aspect(aspect*width/length)
    
    for row in range(length):
        for col in range(width):
            if grid[row][col] is not None:
                subplot.scatter([col], [length-row-1], color='k', s=size)

                if grid[row][col].big is not None:
                    for brow in range(row):
                        for bcol in range(width):
                            if grid[brow][bcol] == grid[row][col].big:
                                if col==bcol:
                                    subplot.plot([col, bcol], [length-row-0.75, length-brow-1.25], color='k',
                                             linestyle='-', linewidth=linewidth)
                                    pass
                                else:
                                    subplot.plot([bcol, bcol], [length-brow-1.25, length-brow-1.5], color='k',
                                             linestyle='-', linewidth=linewidth)
                                    subplot.plot([bcol, col], [length-brow-1.5, length-brow-1.5], color='k',
                                             linestyle='-', linewidth=linewidth)
                                    subplot.plot([col, col], [length-row-0.75, length-brow-1.5], color='k',
                                             linestyle='-', linewidth=linewidth)

def orgTree(grid):
    length=len(grid)

    # Try to optimize the tree
    for row in range(length):
        # Rank what to do first based on size of tree below this node
        grid = expandGrid(grid,2)
        width = len(grid[0])
        priorities = []
        for col in range(width):
            if grid[row][col] is None:
                priorities.append(-1)
            else:
                priorities.append(grid[row][col].calcCount())

        # For each node, try to move littles into the best arrangement
        for col in np.argsort(priorities)[::-1]:   # for each column, looping in order of priority
            if priorities[col]==-1:                 # if it's empty, just leave it and all that follows
                break
            else:
                for little in grid[row][col].littles:
                    found = False
                    for lrow in range(row, length):  # search through the rest of the tree to find the little in question
                        for lcol in range(width):
                            if grid[lrow][lcol] == little:  # when we find one
                                found = True
                                newlcol = col               # try to put it directly beneath the big
                                ii = 0
                                while grid[lrow][lcol] == little:
                                    headroom = [grid[r][newlcol] is None for r in range(row+1, lrow)]
                                    colrange = range(col+1, newlcol+1) if newlcol>col else range(newlcol,col)
                                    runout = [grid[row][c] is None for c in colrange]
                                    
                                    if False in headroom:  # something obstructs the vertical line
                                        newlcol += (-1)**np.random.randint(2)
                                        newlcol = max((0, min(width, newlcol)))
                                    else:  # no obstruction in vertical
                                        if False in runout:  # something obstructs the horizontal line
                                            newlcol += (-1)**np.random.randint(2)
                                            newlcol = max((0, min(width, newlcol)))
                                        else:  # both lines are fine!
                                            if grid[lrow][newlcol] is not None: # something is already there
                                                if little.isSib(grid[lrow][newlcol]):  # if that something is a sibling
                                                    newlcol += (-1)**np.random.randint(2)
                                                    newlcol = max((0, min(width, newlcol)))          # keep walkin
                                                else:  # not related: swap out
                                                    buffer = grid[lrow][newlcol]
                                                    grid[lrow][newlcol] = little
                                                    grid[lrow][lcol] = buffer
                                                    break
                                            else:  # nothing there, all systems go
                                                grid[lrow][newlcol] = little
                                                grid[lrow][lcol] = None
                            if found == True:
                                break
                        if found == True:
                                break

        grid = collapseGrid(grid)
    return grid

def stashLoners(grid):
    loners = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] is not None:
                if grid[row][col].big is None:
                    if grid[row][col].littles == []:
                        loners.append((row, grid[row][col]))
                        grid[row][col] = None
    return grid, loners

def replaceLoners(grid, loners):
    vlines = getVlines(grid)
    newgrid = [row[:] for row in grid]
    for row, loner in loners:
        spaces = np.arange(int(len(grid[0])/4))*4 + 1 + (row%2)*2
        for col in spaces:
            if grid[row][col] is None and not vlines[row][col]:
                grid[row][col] = loner
                break
    return grid
                

def gridEnergy(grid, tension, centering):
    en = 0
    width = len(grid[0])
    length = len(grid)
    for row in range(length):
        for col in range(width):
            if grid[row][col] is not None:
                #en += 0.2 * tension * (col - width/2)**2  # draw towards the middle
                if grid[row][col].big is not None:
                    found = False
                    for br in range(row):
                        for bc in range(width):
                            if grid[br][bc] == grid[row][col].big:
                                found = True
                                en += tension * (bc-col)**2
                            if found:
                                break
                        if found:
                            break
    return en

class spaceNode:
    def __init__(self, row, col, sups):
        self.row = row
        self.col = col
        self.subs = []
        self.sups = []
        if sups is not None:
            for sup in sups:
                if np.abs(sup.col - self.col)<=1:
                    sup.addSub(self)
                    self.sups.append(sup)
    
    def addSub(self, node):
        self.subs.append(node)
    
    def history(self):
        if self.sups == []:
            return [self.col]
        else:
            path = self.sups[np.random.randint(len(self.sups))].history() + [self.col]
            return path
        
def clearSpace(grid, startcols):   # remove a quasi-vertical slice of empty space
    newgrid = [grid[ii][:] for ii in range(len(grid))]
    length = len(newgrid)
    width = len(newgrid[0])
    
    done = False
    while not done:
        fail = False
        if startcols is None:  # generate list of possible starting columns
            startcols = np.arange(width)
        start = np.random.choice(startcols)   # pick one!
        while grid[0][start] is not None:     # if it landed on somebody in the first row...
            startcols = np.delete(startcols,np.argwhere(startcols==start))   # delete it from the available list
            if len(startcols)==0:
                return newgrid, [], [], startcols
            start = np.random.choice(startcols)     # and pick again
        startnode = spaceNode(0,start,None)   # once we have a good one, make a node for it
        nodes = [[startnode]]
        vlines = getVlines(newgrid)   # where are the vertical lines on the grid? (crossing them causes panic and mayhem)
    
        for row in range(1,length):    # for each row
            if nodes[row-1] == []:     # if the row above was empty, then it's a dead end.
                startcols = np.delete(startcols,np.argwhere(startcols==startnode.col))  # remove it from the list
                fail = True  # flag to expedite restarting the while loop
                break  # escape the for loop

            else:                          # assuming there is something to continue building
                nodes.append([])           # build a placeholder for nodes
                stems = []
                branches = []
                for node in nodes[row-1]:  # for each of the nodes in the previous row
                    for step in [-1,0,1]:  # look directly below, and one to each side (hence quasi-vertical)
                        if (node.col+step >= 0) and (node.col+step < width):     # keep it within the width of the grid
                            if newgrid[row][node.col+step] is None and not vlines[row][node.col+step]:   # if that point is neither a bro or a vline
                                stems.append(node.col)  # add those two columns to the lists
                                branches.append(node.col+step)
                branches = set(branches)  # remove duplicates
                stems = set(stems)
                for bcol in branches:   # Now actually build the nodes on the tree (not a simple tree--each node can have <=3 parents)
                    sups = [node for node in nodes[row-1] if node.col in stems]
                    nodes[row].append(spaceNode(row, bcol, sups))
    
        # make one last check to see if things dead-ended on the last row
        if nodes[-1] == [] and not fail:   # if it had failed earlier, though, don't test it again
                startcols = np.delete(startcols,np.argwhere(startcols==startnode.col))
                fail = True  # flag to expedite restarting the while loop
        
        if fail:
            if len(startcols)==0:   # we're plum out of options
                done = True    # we've done the best we can, retire proudly
                indices = []
                nodes = []
                
        
        else:  # if there are percolating quasi-vertical paths
            end = nodes[-1][np.random.randint(len(nodes[-1]))]    # pick one of the bottom row nodes at random
            indices = end.history()     # Walk up through its parents to the top, thus generating the line to be removed
            
            
            for row in range(length):   # cut those points out of the grid
                del newgrid[row][indices[row]]

            startcols = np.delete(startcols,np.argwhere(startcols==startnode.col))   # delete that starting column from the list
            for ii,sc in enumerate(startcols):
                if sc>startnode.col:
                    startcols[ii] -= 1  # shift indices to keep up with the deletion
            
            done = True  # we did what we came here to do, get in the car.
    
    return newgrid, indices, nodes, startcols

def detectConflict(grid):
    for row in range(length):
        for col in range(width):
            if grid[lrow][lcol] is not None:
                while grid[lrow][lcol] == little:
                    headroom = [grid[r][newlcol] is None for r in range(row+1, lrow)]
                    colrange = range(col+1, newlcol+1) if newlcol>col else range(newlcol,col)
                    runout = [grid[row][c] is None for c in colrange]
                    
def getVlines(grid):
    length = len(grid)
    width = len(grid[0])
    vlines = [[False for x in range(width)] for y in range(length)]
    for row in range(length):
        for col in range(width):
            if grid[row][col] is not None:
                bro = grid[row][col]
                if bro.big is not None:
                    gen = (classnum(bro.pmclass) - classnum(bro.big.pmclass))
                    for rr in range(row-gen+1, row):
                        vlines[rr][col] = True      
    return vlines

def countBros(grid):
    length = len(grid)
    width = len(grid[0])
    count = 0
    for row in range(length):
        for col in range(width):
            count += 1 if grid[row][col] is not None else 0
    return count

def getFamilies(grid):
    fams = []
    length = len(grid)
    width = len(grid[0])
    for row in range(length):
        for col in range(width):
            if grid[row][col] is not None:
                bro = grid[row][col]
                if bro.big is None:
                    fams.append([bro])
                    fams[-1] = fams[-1] + getDescendents(bro)
    
    # Now we have the disconnected groups, but I need to sort them left to right
    famsort = []
    for col in range(width):
        thiscol = [grid[ii][col] for ii in range(length) if grid[ii][col] is not None]
        for bro in thiscol:
            for fam in fams:
                if bro in fam and fam not in famsort:
                    famsort.append(fam)
    return famsort

def famWidth(grid, fam):
    stop = 0
    start = len(grid[0])
    for bro in fam:
        _, pos = findBro(grid, bro)
        start = min(start, pos)
        stop = max(stop, pos)
    width = stop-start + 1
    return width, start
                    
def getDescendents(bro):
    if bro.littles == []:
        return []
    else:
        desc = bro.littles[:]
        for little in bro.littles:
            desc += getDescendents(little)
        return desc

def tension(grid):
    tens = 0
    for ii,row in enumerate(grid):
        for jj,bro in enumerate(row):
            if bro is not None:
                if len(bro.littles)>0:  # if this bro has littles
                    for little in bro.littles:   # find horizontal distance to that little
                        _, pos = findBro(grid, little)
                        tens += (pos - jj)**2  # square it and add it to the total tension
    return tens

def energy(grid):
    vac = 0
    for ii,row in enumerate(grid):
        vacs = []
        for seq in groupby(row):
            if seq[0] is None:
                vacs.append(sum(1 for x in seq[1])**2)
        #vac += sum(vacs[1:-1])
        vac += sum(vacs)
    return vac

def jigsaw(grid):
    fams = getFamilies(grid)
    widths = []
    starts = []
    for fam in fams:
        width, start = famWidth(grid, fam) 
        widths.append(width)
        starts.append(start)
        
    order = np.arange(len(fams))
    np.random.shuffle(order)
    
    totalwidth = 0
    space = 10
    newwidth = np.sum(widths) + space*(len(order)-1)
    newgrid = [[None]*newwidth for x in range(len(grid))]
    
    for num, ii in enumerate(order):
        for bro in fams[ii]:
            row, oldcol = findBro(grid, bro)
            newcol = (oldcol - starts[ii]) + totalwidth + space*num
            newgrid[row][newcol] = bro
        totalwidth+= widths[ii]
    return newgrid

def compactify(grid):
    lost = 0
    startcols = np.arange(len(grid[0]))
    while len(startcols)>0:
        grid, indices, nodes, startcols = clearSpace(grid, startcols)
    return grid
        
def hop(grid, bro):
    newgrid = [row[:] for row in grid]
    vlines = getVlines(grid)
    row, oldcol = findBro(grid, bro)
    if bro.big is not None:
        bigrow, _ = findBro(grid, bro.big)
    limits = [oldcol, oldcol]
    for ii, step in enumerate((-1, 1)): # find limits of motion based on space, vlines, headroom and footroom
        col = oldcol
        search = True
            
        while search:
            col += step
            if col>=len(grid[0]) or col<0:
                search = False
                break
                
            if grid[row][col] is not None or vlines[row][col]:
                search = False
                break
            
            if bro.big is not None:
                column = [grid[ii][col] for ii in range(len(grid))]
                headroom = [x==None for x in column[bigrow:row]]
                if False in headroom:
                    if False not in headroom[1:]:  # the only fail is at the level of the big
                        if len(grid[bigrow][col].littles)==0:
                            pass # doesn't impinge on headroom it if has no descendance lines
                        elif grid[bigrow][col] == bro.big:
                            pass  # this should be obvious, but isn't in cases where the big has already hopped off
                        else:
                            search = False
                            break
                    else:
                        search = False
                        break

            if len(bro.littles) > 0:
                below = [grid[ii][col] for ii in range(row+1, len(grid)) if grid[ii][col] is not None]
                for foot in below:
                    if foot.big is not None and foot.big is not bro:
                        footbigrow, _ = findBro(grid, foot.big)
                        if footbigrow == row: # this would cause overlap of horizontal descendent lines
                            search = False
                            break
        
        limits[ii] = col - step


        
    # find force
    relcols = []
    bigprior = 3
    
    if len(bro.littles)>0:
        for little in bro.littles:
            _, relcol = findBro(grid, little)
            relcols.append(relcol)
            
    if bro.big is not None:
        _, relcol = findBro(grid, bro.big)
        relcols.append(relcol*bigprior)
        cog = np.round(np.sum(relcols)/ (len(relcols)+(bigprior-1)))  # center of gravity of this bro's relatives
    else:
        cog = np.round(np.sum(relcols)/ (len(relcols)))  # center of gravity of this bro's relatives

    # move bro to new location
    newcol = int(np.min((limits[1], np.max((limits[0], cog)))))
    newgrid[row][oldcol] = None
    newgrid[row][newcol] = bro
    return newgrid

def tighten(grid):
    newgrid = [row[:] for row in grid]
    allbros = []
    for row in newgrid:
        rowbros = [x for x in row if x is not None]
        np.random.shuffle(rowbros)
        allbros += rowbros
    #np.random.shuffle(allbros)

    for bro in allbros:
        newgrid = hop(newgrid, bro)
    return newgrid

def findBro(grid, bro):
    for ii, row in enumerate(grid):
        if bro in row:
            ncol = row.index(bro)
            nrow = ii
            break
    try:
        return nrow, ncol
    except:
        print(bro)
        
def bestFit(grid, iters=15):
    newgrid = [row[:] for row in grid]
    bestenergy = energy(grid)
    for ii in range(iters):
        test, loners = stashLoners(newgrid)
        test = compactify(tighten(compactify(jigsaw(test))))
        test = replaceLoners(test, loners)
        testenergy = energy(test)
        if testenergy < bestenergy:
            newgrid = [row[:] for row in test]
            bestenergy = testenergy
    return newgrid