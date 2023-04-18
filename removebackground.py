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
                             QFileDialog, QMainWindow, QMessageBox, QTextEdit, QMenu, QFrame, QRadioButton, QSpacerItem, QDoubleSpinBox)
from PyQt5 import QtCore  # conda install pyqt
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from  matplotlib.figure import Figure

#import pySPM as spm
import copy

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import mode
from skimage import filters
from skimage.restoration import rolling_ball
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import sys
import random
import math

import config
import imagedisplay as ImD


class RemovebackgroundWindow(QtWidgets.QWidget):

    def __init__(self,parent=None):

        super(RemovebackgroundWindow, self).__init__(parent)
        
        #.left = 300
        #self.top = 300
        #self.width = 300
        #self.height = 400

       # if config.DispState  == 0:
       #     return 
        
        result = config.get_savedparam("panel", "Remove Background")
        if result is not None:
          # 一致する行が見つかった場合は、resultを処理する
            config.panel_left, config.panel_top, config.panel_width, config.panel_height = result
        else:
            config.panel_width= 300
            config.panel_height = 200
            config.panel_top = 100
            config.panel_left = 100

        self.setGeometry(config.panel_left , config.panel_top , config.panel_width, config.panel_height) 

        self.setWindowTitle("Remove Background")

        self.main_layout = QVBoxLayout()
        
        # Plane group box
        self.plane_group_box = QGroupBox("Plane")
        self.plane_group_box.setFixedSize(250, 150)
        self.plane_layout = QVBoxLayout()
        
        self.plane_off_radio_button = QRadioButton("Off")
        self.plane_off_radio_button.toggled.connect(self.update_rb_plane_type)
        self.plane_layout.addWidget(self.plane_off_radio_button)

        # Create the QHBoxLayout
        self.polynomial_plane_layout = QHBoxLayout()
        self.polynomial_plane_layout.setSpacing(1)  # ここでスペーシングを調整

        # Add the radio button to the layout
        self.polynomial_plane_radio_button = QRadioButton("Polynomial")
        self.polynomial_plane_radio_button.setMinimumHeight(17)
        self.polynomial_plane_radio_button.toggled.connect(self.update_rb_plane_type)        
        self.polynomial_plane_layout.addWidget(self.polynomial_plane_radio_button)


        # Add the label and spin box to the layout      
        self.plane_order_label= QLabel("Order")
        self.plane_order_label.setMinimumHeight(17)
        self.plane_order_spin_box = QSpinBox()
        self.plane_order_spin_box.setMinimumHeight(17)
        self.plane_order_spin_box.setSingleStep(1)
        self.plane_order_spin_box.setRange(1, 5)
        self.plane_order_spin_box.setValue(config.rb_plane_order) 
        self.plane_order_spin_box.valueChanged.connect(self.update_plane_order)
       
        self.plane_order_layout = QHBoxLayout()       
        self.plane_order_layout.addWidget(self.plane_order_label)
        self.plane_order_layout.addWidget(self.plane_order_spin_box)    

        # Add the label and spin box layout to the polynomial layout
        self.polynomial_plane_layout.addLayout(self.plane_order_layout)        
        self.plane_layout.addLayout(self.polynomial_plane_layout)

        ###
        # Create the QHBoxLayout
        self.rolling_layout = QHBoxLayout()
        self.rolling_layout.setSpacing(1)  # ここでスペーシングを調整

        # Add the radio button to the layout
        self.rolling_radio_button = QRadioButton("Rolling")
        self.rolling_radio_button.setMinimumHeight(17)
        self.rolling_radio_button.toggled.connect(self.update_rb_plane_type)        
        self.rolling_layout.addWidget(self.rolling_radio_button)


        # Add the label and spin box to the layout      
        self.rolling_size_label= QLabel("Size (pixels)")
        self.rolling_size_label.setMinimumHeight(17)
        self.rolling_size_spin_box = QSpinBox()
        self.rolling_size_spin_box.setMinimumHeight(17)
        self.rolling_size_spin_box.setSingleStep(1)
        self.rolling_size_spin_box.setRange(1, 500)
        self.rolling_size_spin_box.setValue(config.rb_rolling_size) 
        self.rolling_size_spin_box.valueChanged.connect(self.update_rolling_size)
       
        self.rolling_size_layout = QHBoxLayout()       
        self.rolling_size_layout.addWidget(self.rolling_size_label)
        self.rolling_size_layout.addWidget(self.rolling_size_spin_box)    

        # Add the label and spin box layout to the polynomial layout
        self.rolling_layout.addLayout(self.rolling_size_layout)        
        self.plane_layout.addLayout(self.rolling_layout)

        ####

        # Add the plane_layout to the plane_group_box
        self.plane_group_box.setLayout(self.plane_layout)

        if config.rb_plane_type == 0:
            self.plane_off_radio_button.setChecked(True)
        elif config.rb_plane_type == 1:
            self.polynomial_plane_radio_button.setChecked(True)
        elif config.rb_line_type == 2:
            self.rolling_radio_button.setChecked(True)
        # elif config.rb_line_type == 3:
        #     self.mediandiff_radio_button.setChecked(True)
        # elif config.rb_line_type == 4:
        #     self.mode_radio_button.setChecked(True)   
        # elif config.rb_line_type == 5:
        #     self.histogram_radio_button.setChecked(True)   


        # Line-by-Line group box
        self.line_by_line_group_box = QGroupBox("Line-by-Line")
        self.line_by_line_group_box.setFixedSize(250, 250)
        self.line_by_line_layout = QVBoxLayout()

        # Add combo box for selecting direction
        self.direction_combo_box = QComboBox()
        self.direction_combo_box.addItem("Horizontal")
        self.direction_combo_box.addItem("Vertical")
        self.direction_combo_box.setFixedWidth(150)
        self.direction_combo_box.setCurrentText(config.rb_line_direction)  # rb_line_directionに応じて初期値を設定
        self.direction_combo_box.currentIndexChanged.connect(self.update_direction)
        
        self.line_by_line_layout.addWidget(self.direction_combo_box)
         
        self.line_off_radio_button = QRadioButton("Off")
        self.line_off_radio_button.toggled.connect(self.update_rb_line_type)
        self.line_by_line_layout.addWidget(self.line_off_radio_button)
        
        # Create the QHBoxLayout
        self.polynomial_line_layout = QHBoxLayout()
        self.polynomial_line_layout.setSpacing(1)  # ここでスペーシングを調整

        # Add the radio button to the layout
        self.polynomial_line_radio_button = QRadioButton("Polynomial")
        self.polynomial_line_radio_button.setMinimumHeight(17)
        self.polynomial_line_radio_button.toggled.connect(self.update_rb_line_type)        
        self.polynomial_line_layout.addWidget(self.polynomial_line_radio_button)

        # Add the label and spin box to the layout      
        self.line_order_label= QLabel("Order")
        self.line_order_label.setMinimumHeight(17)
        self.line_order_spin_box = QSpinBox()
        self.line_order_spin_box.setMinimumHeight(17)
        self.line_order_spin_box.setSingleStep(1)
        self.line_order_spin_box.setRange(1, 5)
        self.line_order_spin_box.setValue(config.rb_line_order)  # rb_line_orderに応じて初期値を設定
        self.line_order_spin_box.valueChanged.connect(self.update_line_order)
       
        self.line_order_layout = QHBoxLayout()       
        self.line_order_layout.addWidget(self.line_order_label)
        self.line_order_layout.addWidget(self.line_order_spin_box)    

        # Add the label and spin box layout to the polynomial layout
        self.polynomial_line_layout.addLayout(self.line_order_layout)        
        self.line_by_line_layout.addLayout(self.polynomial_line_layout)
              

        self.median_radio_button = QRadioButton("Median")
        self.median_radio_button.setMinimumHeight(25)
        self.median_radio_button.toggled.connect(self.update_rb_line_type)
        self.line_by_line_layout.addWidget(self.median_radio_button)

        self.mediandiff_radio_button = QRadioButton("Median Difference")
        self.mediandiff_radio_button.setMinimumHeight(25)
        self.mediandiff_radio_button.toggled.connect(self.update_rb_line_type)
        self.line_by_line_layout.addWidget(self.mediandiff_radio_button)

        self.mode_radio_button = QRadioButton("Mode")
        self.mode_radio_button.setMinimumHeight(25)
        self.mode_radio_button.toggled.connect(self.update_rb_line_type)
        self.line_by_line_layout.addWidget(self.mode_radio_button)

         # Create the QHBoxLayout
        self.histogram_layout = QHBoxLayout()
        self.histogram_layout.setSpacing(1)  # ここでスペーシングを

        self.histogram_radio_button = QRadioButton("Histogram")
        self.histogram_radio_button.setMinimumHeight(25)
        self.histogram_radio_button.toggled.connect(self.update_rb_line_type)
        self.histogram_layout.addWidget(self.histogram_radio_button)

        self.threshold_label = QLabel("Type")
        self.threshold_label.setMinimumHeight(17)
        self.threshold_combo_box = QComboBox()
        self.threshold_combo_box.addItem("Manual")
        self.threshold_combo_box.addItem("Otsu")
        self.threshold_combo_box.addItem("Mean")
        self.threshold_combo_box.addItem("Minimum")
        self.threshold_combo_box.setFixedWidth(100)
        self.threshold_combo_box.setMinimumHeight(20)
        self.threshold_combo_box.setStyleSheet("QComboBox { height: 30px; }")
        self.threshold_combo_box.setCurrentText(config.rb_threshold_type)
        self.threshold_combo_box.currentIndexChanged.connect(self.update_threshold_type)

        self.threshold_type_layout = QHBoxLayout()        
        self.threshold_type_layout.addWidget(self.threshold_label)
        self.threshold_type_layout.addWidget(self.threshold_combo_box)

        # Add the label and vombo box layout to the histogram layout
        self.histogram_layout.addLayout(self.threshold_type_layout)        
        self.line_by_line_layout.addLayout(self.histogram_layout)


        self.threshold_value_layout = QHBoxLayout() 
        self.threshold_value_layout.setSpacing(1)  # ここでスペーシングを

        # Add a vertical spacer item to increase the spacing above the threshold_spin_box
        spacer_item = QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.line_by_line_layout.addItem(spacer_item)

    
        self.threshold_label = QLabel("Threshold (%)")
        self.threshold_spin_box = QDoubleSpinBox()
        self.threshold_spin_box.setRange(0.1, 50)
        self.threshold_spin_box.setSingleStep(0.1)
        self.threshold_spin_box.setValue(config.rb_threshold)
        self.threshold_spin_box.setFixedWidth(80)
        self.threshold_spin_box.valueChanged.connect(self.update_threshold)
        self.threshold_value_layout.addWidget(self.threshold_label)
        self.threshold_value_layout.addWidget(self.threshold_spin_box)
        self.line_by_line_layout.addLayout(self.threshold_value_layout)

        # Add the line_by_line_group to the line-by-line layout
        self.line_by_line_group_box.setLayout(self.line_by_line_layout)
        
        if config.rb_line_type == 0:
            self.line_off_radio_button.setChecked(True)
        elif config.rb_line_type == 1:
            self.polynomial_radio_button.setChecked(True)
        elif config.rb_line_type == 2:
            self.median_radio_button.setChecked(True)
        elif config.rb_line_type == 3:
            self.mediandiff_radio_button.setChecked(True)
        elif config.rb_line_type == 4:
            self.mode_radio_button.setChecked(True)   
        elif config.rb_line_type == 5:
            self.histogram_radio_button.setChecked(True)    
        
        self.main_layout.addWidget(self.plane_group_box)
        self.main_layout.addWidget(self.line_by_line_group_box)

        self.setLayout(self.main_layout)

   
    def update_threshold(self):

        config.rb_threshold  = self.threshold_spin_box.value()

        disp = ImD.ImageDisplay()
        disp.DispAryData()

    def update_rb_line_type(self):
        if self.line_off_radio_button.isChecked():
            config.rb_line_type = 0
        elif self.polynomial_line_radio_button.isChecked():
            config.rb_line_type = 1
        elif self.median_radio_button.isChecked():
            config.rb_line_type = 2
        elif self.mediandiff_radio_button.isChecked():
            config.rb_line_type = 3
        elif self.mode_radio_button.isChecked():
            config.rb_line_type = 4
        elif self.histogram_radio_button.isChecked():
            config.rb_line_type = 5
       # elif self.facet_radio_button.isChecked():
       #     config.rb_line_type = 4
       # elif self.histogram_radio_button.isChecked():
       #     config.rb_line_type = 5

     #test
        disp = ImD.ImageDisplay()
        disp.DispAryData()

    def update_direction(self, index):
        if index == 0:
            config.rb_line_direction= "Horizontal"
        elif index == 1:
            config.rb_line_direction = "Vertical"
        # Update other functions or variables based on the selected direction

        disp = ImD.ImageDisplay()
        disp.DispAryData()
    
    def update_rb_plane_type(self):
        if self.plane_off_radio_button.isChecked():
            config.rb_plane_type = 0
        elif self.polynomial_plane_radio_button.isChecked():
            config.rb_plane_type = 1
        elif self.rolling_radio_button.isChecked():
            config.rb_plane_type = 2
        # elif self.mediandiff_radio_button.isChecked():
        #     config.rb_line_type = 3
        # elif self.mode_radio_button.isChecked():
        #     config.rb_line_type = 4
        # elif self.histogram_radio_button.isChecked():
        #     config.rb_line_type = 5
       # elif self.facet_radio_button.isChecked():
       #     config.rb_line_type = 4
       # elif self.histogram_radio_button.isChecked():
       #     config.rb_line_type = 5

     #test
        disp = ImD.ImageDisplay()
        disp.DispAryData()

    def update_plane_order(self):

        config.rb_plane_order = self.plane_order_spin_box.value()

        disp = ImD.ImageDisplay()
        disp.DispAryData()
    
    def update_rolling_size(self):

        config.rb_rolling_size = self.rolling_size_spin_box.value()

        disp = ImD.ImageDisplay()
        disp.DispAryData()
    
    def update_line_order(self):

        config.rb_line_order = self.line_order_spin_box.value()

        disp = ImD.ImageDisplay()
        disp.DispAryData()
    
    def update_threshold_type(self, index):

        if index == 0:
            config.rb_threshold_type= "Manual"
        elif index == 1:
            config.rb_threshold_type = "Otsu"
        elif index == 2:
            config.rb_threshold_type = "Mean"
        elif index == 3:
            config.rb_threshold_type = "Minimum"


        # Update other functions or variables based on the selected direction

        disp = ImD.ImageDisplay()
        disp.DispAryData()

def polynomial_line(tempdata):

    fitdata = np.zeros_like(tempdata)  # 全ての要素が0の配列を作成する（arraydataと同じ形状）

    for i, row in enumerate(tempdata):
        x = np.arange(len(row))
        y = row
        coeffs = np.polyfit(x, y, config.rb_line_order)  # order次の最小二乗フィッティングを行う
        fitdata[i] =np.polyval(coeffs, x)  # フィッティング曲線のy値を計算し、fitdataに格納する
    
    return fitdata
    
def median_line(tempdata):

    #二次元配列arraydataの各行ごとに、各列のデータの中央値を二次元配列にして戻す関数。
    fitdata = np.zeros((tempdata.shape[0], tempdata.shape[1]))
    median_all = np.median(tempdata)

    for i, row in enumerate(tempdata):
        median = np.median(row)  # 各行の中央値を計算する

        fitdata[i] = np.full(tempdata.shape[1], median)  # 中央値を各列に入れる

    return fitdata
    
def mediandiff_line(tempdata, inline=True):
    """
    Correct the image with the median difference
    """
    N = tempdata
    # Difference of the pixel between two consecutive rows
    N2 = N - np.vstack([N[:1, :], N[:-1, :]])
    # Take the median of the difference and cumsum them
    C = np.cumsum(np.median(N2, axis=1))
    # Extend the vector to a matrix (row copy)
    D = np.tile(C, (N.shape[0], 1)).T

    if inline:
        return D
    else:
        New = copy.deepcopy(data)
        return D
 
    # DataIOオブジェクトに変換する
    # data = px.io.data_io.DataIO("array")
    # # データを設定する
    # data.data = tempdata
    
    # # データの前処理
    # data = px.Processing.process(data, process_name='Gridding', verbose=False)

    return fitdata_tiled 

def histogram_line(tempdata):

    fitdata = np.zeros_like(tempdata)  # 全ての要素が0の配列を作成する（arraydataと同じ形状）
    thresholod = 0
    if(config.rb_threshold_type == "Manual"):

         # Compute the threshold using Otsu's algorithm

        min_val = tempdata.min()
        max_val = tempdata.max()
        
        threshold = config.rb_threshold/100*(max_val-min_val)+min_val

    if(config.rb_threshold_type == "Otsu"):

         # Compute the threshold using Otsu's algorithm
        threshold = filters.threshold_otsu(tempdata)
    
    elif(config.rb_threshold_type == "Mean"):

          # Compute the threshold using Mean algorithm
        threshold = filters.threshold_mean(tempdata)

    elif(config.rb_threshold_type == "Minimum"):

          # Compute the threshold using Minimum algorithm
        threshold = filters.threshold_minimum(tempdata)

        # Create a copy of tempdata
    new_arraydata = tempdata.copy()

        # Set the values in new_arraydata greater than the threshold to NaN
    new_arraydata[tempdata > threshold] = threshold 

        # for i, row in enumerate(new_arraydata):
        #     row_mode = mode(row)  # 各行の最頻値を計算する
        #     fitdata[i] = np.full(tempdata.shape[1], row_mode.mode)  # 最頻値を各列に入れる

    for i, row in enumerate(new_arraydata):
        x = np.arange(len(row))
        y = row
        coeffs = np.polyfit(x, y, 1)  # order次の最小二乗フィッティングを行う
        fitdata[i] =np.polyval(coeffs, x)  # フィッティング曲線のy値を計算し、fitdataに格納する
   
    
    return fitdata #new_arraydata
    
def mode_line(tempdata):

     # 二次元配列arraydataの各行ごとに、各列のデータの最頻値を二次元配列にして戻す関数。
    fitdata = np.zeros((tempdata.shape[0], tempdata.shape[1]))
    mode_all = mode(tempdata, axis=None)  # 全体の最頻値を計算する
    for i, row in enumerate(tempdata):
        row_mode = mode(row)  # 各行の最頻値を計算する
        fitdata[i] = np.full(tempdata.shape[1], row_mode.mode)  # 最頻値を各列に入れる
    return fitdata
   
def normal_vectors(image, sx, sy):
    normal = np.zeros((*image.shape, 3), dtype=float)
    normal[:, :, 0] = -sx
    normal[:, :, 1] = -sy
    normal[:, :, 2] = 1
    normal /= np.sqrt(np.sum(normal**2, axis=2))[:, :, np.newaxis]
    return normal

def facet_leveling(tempdata, iterations=20, c=1/20):
    # Gradient計算
    sx = ndimage.sobel(tempdata, axis=0)
    sy = ndimage.sobel(tempdata, axis=1)

    # Normalベクトル計算
    normals = normal_vectors(tempdata, sx, sy)

    for _ in range(iterations):
        # ローカル法線の共分散行列を計算
        cov_matrix = np.average(np.einsum('...i,...j->...ij', normals, normals), axis=(0, 1))

        # 固有値・固有ベクトルを計算
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 最も大きい固有値に対応する固有ベクトルが主要な法線ベクトル
        prevalent_normal = eigenvectors[:, np.argmax(eigenvalues)]

        # 面の法線ベクトルに対応する平面を求める
        plane = prevalent_normal[0] * sx + prevalent_normal[1] * sy + prevalent_normal[2] * tempdata

        # 求めた平面を元の画像から引く
        #image -= plane

    return plane


def Removebackrgoud_Plane():

    #config.ZaryData = cv2.blur(config.RawaryData, ksize=(config.kernel_size , config.kernel_size ))  
    if config.rb_plane_type !=0 :

        if config.rb_plane_type == 1: # Polynomianl fitting
            # 最小二乗フィッティングの準備
            rows, cols = config.ZaryData.shape
            x, y = np.meshgrid(np.arange(cols), np.arange(rows))

            # 特徴行列Xを作成
            X = np.column_stack([x.ravel()**i * y.ravel()**j for i in range(config.rb_plane_order+1) for j in range(config.rb_plane_order+1) if i+j <= config.rb_plane_order])
            Y = config.RawaryData.ravel()

             # 最小二乗フィッティング
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

             # 同じ次元の二次元配列にフィッティング結果を適用
            config.ZaryData = config.RawaryData-(X @ coeffs).reshape(rows, cols)

        elif config.rb_plane_type == 2: # Rolling fitting

            # Smooth the data using a Gaussian filter
           #smooth_data = gaussian(config.RawaryData, sigma=10)
            # 最小二乗フィッティングの準備
            background = rolling_ball(config.RawaryData, radius=config.rb_rolling_size)

            # Subtract the background from the data
            config.ZaryData = config.RawaryData-background



def Removebackrgoud_Line():

    rb = RemovebackgroundWindow      
        
    fitdata = np.zeros_like(config.ZaryData)  # 全ての要素が0の配列を作成する（arraydataと同じ形状）
    tempdata = np.zeros_like(config.ZaryData)

    if config.rb_line_type !=0 :
                
        if config.rb_line_direction == "Horizontal":
                    
            tempdata = np.array(config.ZaryData)
                
        elif config.rb_line_direction == "Vertical":
                    
            tempdata = np.array(config.ZaryData.T)

        if config.rb_line_type == 1: # Polynomianl fitting

            fitdata = polynomial_line(tempdata)

            # 各行に対して最小二乗フィッティングを行い、結果を格納する新しい配列を作成する
                   
        elif config.rb_line_type == 2: # median
            fitdata = median_line(tempdata)
        
        elif config.rb_line_type == 3: # median diffference

            fitdata_pre = mediandiff_line(tempdata) # 各行ごとの最頻値を計算する
            fitdata = np.tile(fitdata_pre, (1, tempdata.shape[1]//fitdata.shape[1]+1))[:, :tempdata.shape[1]] # 水平方向に繰り返し
            
        elif config.rb_line_type == 4: # mode

            fitdata = mode_line(tempdata)
        
        elif config.rb_line_type == 5: # histogram

            fitdata = histogram_line(tempdata)


        tempdata -= fitdata

        if config.rb_line_direction == "Horizontal":
                    
            config.ZaryData =tempdata
            
        elif config.rb_line_direction == "Vertical":

            config.ZaryData =tempdata.T
  
