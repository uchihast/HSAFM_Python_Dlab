#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Uchihashi
#
# Created:     23/02/2018
# Copyright:   (c) Uchihashi 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#from guimain import MainWidget
import config
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
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
import noisefilter as nf
import removebackground as rb

class ImageDisplay:

    def __init__(self): # 初期化： インスタンス作成時に自動的に呼ばれる
        config.DispState = 1

    
    def DispAryData(self):

        #config.aryData = config.ZaryData 

        config.ZaryData = np.array(config.RawaryData)
        
        if config.rb_plane_type != 0:
            rb.Removebackrgoud_Plane()

        if config.rb_line_type != 0:
            rb.Removebackrgoud_Line()

        if config.nf_remove_scar_auto == 1:
            nf.RemovceScar()

        if config.nf_auto == 1:
            nf.AutoNoiseFilter()

        config.aryData = np.array(config.ZaryData) #config.ZaryData

        aryfmin = np.min(config.aryData)
        aryfmax= np.max(config.aryData)
        config.aryData = config.aryData-aryfmin
        aryzmin = np.min(config.ZaryData)
        #aryzmax= np.max(config.ZaryData)
        config.ZaryData=config.ZaryData-aryzmin
        #print(config.ZaryData.shape)
        #print(config.ZaryData)
        

        diff = aryfmax-aryfmin
        if(diff == 0):
            diff = 1
        config.aryData = config.aryData/diff
        config.aryData =config.aryData*255 #convert to 0-255scale 

        pilImg = Image.fromarray(np.uint8(config.aryData))
        #cvimg=np.asarray(pilImg)
        cvimg=np.asarray(pilImg)

        #if(config.index  == 0):
        orgHeight, orgWidth = cvimg.shape[:2]
        if(orgWidth < 400):
            config.dspsize =  (400, int(400*orgHeight/orgWidth))

        else:
            config.dspsize =  (orgWidth, orgHeight)

        #print(int(200*orgHeight/orgWidth))
        #print(200*orgHeight/orgWidth)
        #config.dspimg= np.zeros((orgWidth*5,orgHeight*5,  3), np.uint8)
        config.dspimg = np.array([config.YPixel, config.XPixel,3])
        #config.ZaryData= cv2.resize(config.ZaryData,config.dspsize)
        #print(config.ZaryData.shape)
        config.aryData = cv2.resize(config.aryData, config.dspsize)

        config.dspimg  =cv2.applyColorMap(cv2.flip( cv2.resize(cvimg, config.dspsize),0), config.DIcolor)
        
        #config.dspimg = config.dspimg, config.DIcolor);
        
        
        cv2.namedWindow("img1ch")
        #cv2.namedWindow("img1ch", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
        #cv2.imread(config.dspimg)
        cv2.imshow("img1ch", config.dspimg)

        result = config.get_savedparam("panel","img1ch")
        if result is not None:
          # 一致する行が見つかった場合は、resultを処理する
           config.panel_left, config.panel_top, config.panel_width, config.panel_height = result
        else:
            config.panel_top = 10
            config.panel_left = 10

        cv2.moveWindow("img1ch", config.panel_left, config.panel_top)
        #print(config.ZaryData.shape)
        #print(config.ZaryData[0][0])
        #print(np.max(config.aryData))
        
       
      

        
        
