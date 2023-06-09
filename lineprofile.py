# -------------------------------------------------------------------------------
# Name:        Line Profile
# Purpose:
#
# Author:      Uchihashi
#
# Created:     01/03/2018
# Last Edited: 06/13/2021 by shaoma
# Copyright:   (c) Uchihashi 2018
# Licence:     <your licence>
# -------------------------------------------------------------------------------
import config
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QWidget, QApplication, QPushButton, QLineEdit,
                             QHBoxLayout, QVBoxLayout, QTextEdit, QProgressBar,
                             QFileDialog, QListView, QAbstractItemView, QComboBox,
                             QDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QProgressDialog, QPushButton, QSizePolicy, QTableWidget,
                             QTableWidgetItem, QSlider, QSpinBox, QToolButton, QStyle,
                             QCheckBox, QGroupBox, QBoxLayout, QMessageBox, QAction,
                             QFileDialog, QMainWindow, QMessageBox, QTextEdit, QMenu, QFrame, )
from PyQt5 import QtCore  # conda install pyqt
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import CheckButtons, Button
from  matplotlib.figure import Figure
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import math

#plt.ion()
class LineWindow(QMainWindow):

    def __init__(self, parent=None):
        super(LineWindow, self).__init__(parent)
    

def temp_function(arg1,arg2):
    print("tmp_function out of class")
    print("point 0 is :"+str(arg1))
    print("point 1 is :"+str(arg2))

def calc_PLdist(new_pnt,old_line):
   # arg1, new_point: new point [x,y]
   # arg2, old line : existed point set, LineList
   tmp_dist_sub = old_line - new_pnt
   tmp_dist_sqr = tmp_dist_sub*tmp_dist_sub
   tmp_dist_sqr_sum = np.sum(tmp_dist_sqr,axis=1) 
   tmp_dist = np.sqrt(tmp_dist_sqr_sum)
   return tmp_dist
    
# detect radius: if new clicked point is close to the existed profile line by detect_radius
detect_radius = 10


class Node:
    def __init__(self, imageindex = None):
        self.imageindex = imageindex
        self.nextindex = None
    def __array__(self, imageindex) -> np.array:
        return np.array([imageindex[0],imageindex[1]],np.int32)

class index_list:
    def __init__(self):
        self.start_point = None
        self.end_point = None

    #insert new index after middle_node
    def insert_index(self,middle_node, new_index):
        if middle_node is None:
            print("The mentioned node is absent")
            return
        NewNode = Node(new_index)
        NewNode.nextindex = middle_node.nextindex
        middle_node.nextindex = NewNode
    
    def print_index(self):
        printval = self.start_point
        while printval is not None:
            print("Class print:", printval.imageindex)
            printval = printval.nextindex

class LineProfile:

    def __init__(self):
       
       
        self.sx=0
        self.sy=int(config.dspimg.shape[0] / 2)
        self.ex=int(config.dspimg.shape[1])
        self.ey=int(config.dspimg.shape[0] / 2)
        self.LineList=np.empty((2,2),dtype=np.int16)
        self.pts = np.array([[self.sx,self.sy],[self.ex,self.ey]],np.int32)
        self.plot=False
        
       
        self.colorarray=config.dspimg #colored img array
        self.grayarray=cv2.cvtColor(config.dspimg, cv2.COLOR_BGR2GRAY) #gray scale img array from 0 to 255 value
        self.zarray=config.ZaryData
        self.xscansize=config.XScanSize
        self.yscansize=config.YScanSize
        self.xpixel=config.XPixel
        self.ypixel=config.YPixel
        
        self.sxnm=0
        self.synm=0
        self.exnm=0
        self.eynm=0

        self.index=config.index
       
        self.s0=False #click event
        self.s1=False #drag and draw line event
        self.s2=False #drag and update lineprofile event
        self.s3=False #relase event, line existence
        self.s4=False #set start point
        self.s5=False #set end point
        self.status = 0x00
        self.index_list = index_list()


    def CloseEvent(self,event):
        config.lineopen=False
        config.lineclose=True
        
        print("line close event fired")
        
        print("line close"+str(config.lineclose))
        print("line open"+str(config.lineopen))
       
   # draw line between clicked point and fetch z-data along the line, plot z data on new window 
   # 1. detect mouse click event
   # 2a. check if line exists, if no, create line on mouse move
   # 2b. if yes, check if new click close to any existed point or line
   # 2b1. if close to any of the node, drage the node
   # 2b2. if not, create a new node
   # 3. get the pixel index along the line
   # 4. fetch the z-data from given pixel index
   # 5. put the z-data into plot
    def DrawLine(self,event, x, y, flags, param):
      
        if config.lineopen==True and config.lineclose==False:
            
            if event == cv2.EVENT_LBUTTONDOWN:  #左クリック押下時
                #set_startpnt = False
                #set_endpnt = False
                if self.s3:
                    #status s3: released event, line exists
                    #if line exsited, check distance between the clicked point and the line
                    new_pts = [x,y]
                    print("3rd pnt is :",new_pts)
                    tmp_dist = calc_PLdist(new_pts,self.LineList)
                    print("Min Dist is: ",tmp_dist.min())
                    print("Max Dist is: ",tmp_dist.max())
                    print("tmp_dist[0]:",tmp_dist[0])
                    print("tmp_dist[-1]:",tmp_dist[-1])
                    if tmp_dist.min()<detect_radius:
                        if tmp_dist.min() in tmp_dist[0:int(0.10*len(tmp_dist))]:
                            print("min at start point")
                            #set_startpnt = True
                            self.s4 = True
                            self.s5 = False
                            self.sx = x
                            self.sy = y
                            self.index_list.start_point = Node([x,y])
                        elif tmp_dist.min() in tmp_dist[-1:-int(0.10*len(tmp_dist)):-1]:
                            print("min at end point")
                            #set_endpnt = True
                            self.s5 = True
                            self.s4 = False
                            self.ex = x
                            self.ey = y
                            self.index_list.end_point = Node([x,y])
                        else:
                            print("we will add one point here")
                            # fetch closest pixel coordinates in the line
                            tmp_idx = np.where(tmp_dist == tmp_dist.min())[0]
                            self.pts = np.insert([[self.sx,self.sy],[self.ex,self.ey]],1,[x,y],axis=0)
                            print("points on line:",self.pts)
                            print("tmp_idx: ", tmp_idx[0])
                            tmp_idx = tmp_idx[0]
                            tmp_x=self.LineList[tmp_idx,:]
                            print("tmp_x:", tmp_x)
                            print("point list:", self.pts.shape)
                            cv2.circle(self.temp,(tmp_x[0],tmp_x[1]),5,(200,200,0),-1)
                            cv2.imshow('img1ch', self.temp)
                            cv2.waitKey(1)

                    imageH = self.zarray.shape[0]
                    imageW = self.zarray.shape[1]
                    self.s0 = True
                    self.sx = x      #始点x
                    self.sy = y      #始点y
                    self.index_list.start_point = Node([x,y])
                    self.index_list.print_index()
                    self.height_startpoint=self.zarray[self.sy,self.sx]
                else:
                    self.status = self.status or 0x01 
                    imageH = self.zarray.shape[0]
                    imageW = self.zarray.shape[1]
                    print("drawline "+str(self.index))
                    self.s0 = True
                    self.s5 = True
                    self.s4 = False
                    self.sx = x      #始点x
                    self.sy = y      #始点y
                    self.index_list.start_point = Node([x,y])
                    self.index_list.print_index()
                    self.height_startpoint=self.zarray[self.sy,self.sx]

                print("Left-clicked. Start point:(%d, %d)" %(self.sx, self.sy))
                #temp_function(self.sx,self.sy)
                
                
                self.sxnm=(self.sx/self.zarray.shape[1])*self.xscansize
                self.synm=(self.sy/self.zarray.shape[0])*self.yscansize
               
                    
            elif event == cv2.EVENT_MOUSEMOVE and self.s0 and (not self.s1):  #左クリックでドラッグ中
                # if click around start point
                if self.s4:
                    self.sx = x
                    self.sy = y
                    self.index_list.start_point = Node([x,y])
                else:
                    self.ex = x
                    self.ey = y
                    self.index_list.end_point = Node([x,y])
                self.temp=config.dspimg.copy()
                # change to cv2.polylines from line for multipoint function
                #pts = np.array([[self.sx,self.sy],[self.ex,self.ey]],np.int32)
                pts = np.array([self.index_list.start_point.imageindex, self.index_list.end_point.imageindex])
                cv2.polylines(self.temp,[pts],False,(0,255,0),2)
                line_list = np.array([[self.sx,self.sy],[self.ex, self.ey]],dtype=np.int16)
                cv2.circle(self.temp,tuple(line_list[0,:]),detect_radius,(200,200,0),2)
                cv2.circle(self.temp,tuple(line_list[1,:]),detect_radius,(200,200,0),2)
                cv2.imshow('img1ch', self.temp)
                self.s1 = True
                cv2.waitKey(1)
                self.s1 = False

                self.Calculate_Distance()

                dX = self.ex-self.sx
                dY = self.ey-self.sy
                dXa = np.abs(dX)
                dYa = np.abs(dY)

               
                # distancelist[x_px_index, y_px_index, z_value]
                distancelist = np.empty(shape=(np.maximum(dYa, dXa),3),dtype=np.float32)
                distancelist.fill(np.nan)

                negativeY = self.sy > self.ey
                negativeX = self.sx > self.ex

                if self.sx == self.ex and self.sy != self.ey: # vertical line
                    distancelist[:,0] = self.sx
                    if negativeY:
                        distancelist[:,1] = np.arange(self.sy-1, self.sy-dYa-1, -1)
                    else:
                        distancelist[:,1] = np.arange(self.sy+1, self.sy+dYa+1)
                elif self.sy == self.ey and self.sx != self.ex: # horizontal line
                    distancelist[:,1] = self.sy
                    if negativeX:
                        distancelist[:,0] = np.arange(self.sx-1, self.sx-dXa-1, -1)
                    else:
                        distancelist[:,0] = np.arange(self.sx+1, self.sx+dXa+1)
                elif self.sx == self.ex and self.sy == self.ey:
                    pass
                else:
                    steepSlope = dYa > dXa
                    if steepSlope:
                        slope_f32 = np.float32(dX)/np.float32(dY)
                        if negativeY:
                            distancelist[:,1] = np.arange(self.sy-1, self.sy-dYa-1, -1)
                        else:
                            distancelist[:,1] = np.arange(self.sy+1, self.sy+dYa+1)
                        distancelist[:,0] = (slope_f32*(distancelist[:,1]-self.sy)).astype(int)+self.sx
                    
                    else:
                        slope_f32 = np.float32(dY)/np.float32(dX)
                        if negativeX:
                            distancelist[:,0] = np.arange(self.sx-1, self.sx-dXa-1, -1)
                        else:
                            distancelist[:,0] = np.arange(self.sx+1, self.sx+dXa+1)
                        distancelist[:,1] = (slope_f32*(distancelist[:,0]-self.sx)).astype(int)+self.sy

                    colX = distancelist[:,0]
                    colY = distancelist[:,1]
                    distancelist = distancelist[(colX >=0) & (colY >= 0) & (colX < self.zarray.shape[1]) & (colY < self.zarray.shape[0])]
                    distancelist[:,2] = self.zarray[config.dspsize[1]-distancelist[:,1].astype(np.uint),distancelist[:,0].astype(np.uint)]
                    distancelist[:,2] = distancelist[:,2]-distancelist[:,2].min()
                    self.LineList=np.array(distancelist[:,0:2])
                    step=self.distance/len(distancelist[:,2])
                    distance_ticks= np.arange(len(distancelist[:,2]))*step
                    self.UpdatePlot(config.linewindow,config.figure,config.axes,distance_ticks, distancelist)

                self.s2 = True
                        
            
            elif event == cv2.EVENT_LBUTTONUP: #clear the state
                self.s2=False
                self.s1=False
                self.s0=False             
                # if one line exist, s3 will be True until RESET
                self.s3=True 
                self.s4=False
                self.s5=False
                print("Left-releasec. End point:(%d, %d)" %(self.ex, self.ey))
                

               

    def UpdatePlot(self,profile_window,profile_figure,profile_axes,distance_ticks,profile_zvaluelist):
        self.zvaluelist=profile_zvaluelist[:,2]
        self.line=profile_window
        self.figure=profile_figure
        self.axes=profile_axes

        self.line.set_data(distance_ticks,self.zvaluelist)
        self.axes.set_xlim(distance_ticks.min(),distance_ticks.max()) 
        self.axes.set_ylim(self.zvaluelist.min(),self.zvaluelist.max())
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.canvas.mpl_connect("close_event",self.CloseEvent)
    
    def MakeProfileData(self):
        self.dbg_echo()
        #wait to move lineprofile data calculation here
    
    def MakeProfileWindow(self):
        plt.ion()
        self.figure,self.axes= plt.subplots(figsize=(8,6))
        self.fnum=plt.figure(1)
        self.figure.canvas.setWindowTitle("Line Profile")
        Multi_chk_pos = plt.axes([.12, .85, .1, .15]) # (x1,y1,x2,y2): make checkbox at (x1,y1) with a size of (x2,y2)
        Multi_chk_pos.set_axis_off() # turn off the surrounding line of checkbox
        Cursor_chk_pos = plt.axes([.25, .85, .10, .15])
        Cursor_chk_pos.set_axis_off()
        self.Multipnt_chk = CheckButtons(Multi_chk_pos,['Multi-Point'])
        self.Cursor_chk = CheckButtons(Cursor_chk_pos,['Cursor']) # default status is False
        axReset = plt.axes([0.38, 0.9, 0.1, 0.05])
        self.bnext = Button(axReset, 'Reset',color = "white")
        self.bnext.on_clicked(self.dbg_echo)
        self.bnext.on_clicked(self.reset_LineProfile)
        self.line=self.axes.plot([],[])[0]
        config.linewindow=self.line
        config.figure=self.figure
        self.axes.set_xlabel("nm")
        config.axes=self.axes
        self.figure.canvas.mpl_connect("close_event",self.CloseEvent)
    
   

    def MouseSet(self,input_img_name):
        self.input_img_name=input_img_name
        cv2.setMouseCallback(self.input_img_name, self.DrawLine, None)

    # Calculate_Distance function calculates the distance between start point and end point, unit in nm
    # The calculated distance, self.distance is for make linespan of lineprofile plot
    def Calculate_Distance(self):
        self.height_endpoint=self.zarray[self.ey,self.ex]               
        self.exnm=(self.ex/self.zarray.shape[1])*self.xscansize
        self.eynm=(self.ey/self.zarray.shape[0])*self.yscansize
        self.distance=int(np.sqrt((self.exnm - self.sxnm) ** 2 + (self.eynm - self.synm) ** 2))
    
    def pnt_line_dist(self,pts):
        old_line = [[self.sx, self.sy],[self.ex,self.ey]]
        tmp = old_line-pts
    
    #reset_LineProfile just remove the drawn line on image but keep the z data until new line drawn
    def reset_LineProfile(self,event):
        self.temp=config.dspimg.copy()
        cv2.imshow('img1ch', self.temp)
        cv2.waitKey(1)

    def dbg_echo(self,event):
        print("hello!")

    def point_insert(self):
        print("greetings")

    class Node:
        def __init__(self, imageindex = None):
            self.imageindex = imageindex
            self.nextindex = None
    class index_manange:
        def __init__(self):
            self.start_point = None
        
        def print_index(self):
            printval = self.start_point
            while printval is not None:
                print(printval.imageindex)
                printval = printval.nextindex