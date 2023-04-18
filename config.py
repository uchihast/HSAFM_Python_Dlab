#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Uchihashi
#
# Created:     22/02/2018
# Copyright:   (c) Uchihashi 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys


# information of desktop
app = QApplication(sys.argv)
screen = app.primaryScreen()
size = screen.size()
dspH = size.height()
dspW = size.width()

#  panel size definition
panel_width = 300
panel_height = 400
panel_top =10
panel_left =30

FileType = None
FileHeaderSize = None
FrameHeaderSize = None
TextEncoding = None
OpeNameSize  = None
CommentSize  = None
DataType1ch = None
DataType2ch = None
FrameNum = 500
XPixel = None
YPixel = None
XScanSize = None
YScanSize = None
FrameTime= float("nan")
PiezoConstX= float("nan")
PiezoConstY= float("nan")
PiezoConstZ= float("nan")
DriverGainZ= float("nan")
Offset = None
ADRange = None
ADResolution = None
AveFlag = None
AveNum = None
Year = None
Day = None
Hour = None
Minute = None
Second = None
XRound = None
YRound = None
MaxScanSizeX = float("nan")
MaxScanSizeY = float("nan")
FrameNum = None
MachineNo = None
ScanTryNum = None
Sensitivity = float("nan")
PhaseSens = float("nan")
ScanDirection = None
OpeName = ""
Comment = ""

CurrentNum = None
MaxData = None
MiniData = None
XOffset = None
YOffset  = None
XTilt  = None
YTilt = None
LaserFlag = None

files =[]
aryData = np.empty(0)
ZaryData=np.empty(0)
RawaryData=np.empty(0)

row=0

dspsize=1
dspimg =np.empty(0)

DIcolor = np.zeros((256, 1, 3), dtype=np.uint8)

FileNum =0
DispState = 0
DispMode = 1
pbSpeed = 1

# file open parameters
data_folder = "/Volumes/Lacie 16TB/AFM Data"

# Remove bacgorund parameters
rb_plane_type = 0
rb_plane_order = 1
rb_rolling_size = 10
rb_line_order = 1
rb_line_type = 0
rb_line_direction = "Horizontal"
rb_threshold_type = "Manual"
rb_threshold = 30

# Noise Filter parameters
nf_remove_scar_auto = 0
nf_scar_thresh = 0.01
nf_auto = 0
nf_type = "Average"    
nf_kernel_size = 3
nf_sigma_x = 0.1
nf_sigma_y = 0.1
nf_sigma_d = 1 
nf_sigma_space = 50
nf_sigma_color = 75


lineclose=False
lineopen=False

linewindow=None
figure=None
axes=None

def save_params(type, name, variable):
    with open("FalconViewer.parm", "r+") as file:
        
        lines = file.readlines()
        file.seek(0)
        for line in lines:
            data = line.strip().split(",")
            if data[0] == type and data[1] == name :
                if type == "panel":
                    # 一致する行が見つかった場合は、3, 4, 5, 6列目に値を書き込む
                    file.write(f"{type},{name},{panel_left},{panel_top},{panel_width},{panel_height}\n")
                elif type =="param" :
                    file.write(f"{type},{name},{variable}\n")

                        
            else:
                # 一致しない場合は、そのまま書き込む
                file.write(line)
        file.truncate()

def get_savedparam(type, name):
    with open("FalconViewer.parm", "r") as file:
        data = None  # デフォルト値を設定
        for line in file:
            # 行をカンマで分割して、最初の要素がtypeと一致するかをチェックする
            if line.strip().split(",")[0] == type:
                if type == "panel":
                    # 2番目の要素がnameと一致するかをチェックする
                    if line.strip().split(",")[1] == name:
                        # 3番目の要素を返す
                        data = line.strip().split(",")
                        return [int(data[i]) for i in range(2, 6)]

                if type == "param":
                    # 2番目の要素がnameと一致するかをチェックする
                    if line.strip().split(",")[1] == name:
                        # 3番目の要素を返す
                        data = line.strip().split(",")
                        return data[2]

        # 一致する行が見つからない場合は、Noneを返す
        return None
