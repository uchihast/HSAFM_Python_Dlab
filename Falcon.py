# -------------------------------------------------------------------------------
# Name:        Falcon.py
# Purpose:
#
# Author:      Uchihashi
#
# Created:     27/02/2018
# Copyright:   (c) Uchihashi 2018
# Licence:     <your licence>
# -------------------------------------------------------------------------------

import sys
import os
import math
import struct
from PyQt5.QtCore import (QDir, QIODevice, QFile, QFileInfo, Qt, QTextStream,
                          QUrl, QSize, pyqtSignal, QMutexLocker, QMutex, QThread, QSettings, Qt)

from PyQt5.QtWidgets import (QWidget, QApplication, QPushButton, QLineEdit,
                             QHBoxLayout, QVBoxLayout, QTextEdit, QProgressBar,
                             QFileDialog, QListView, QAbstractItemView, QComboBox,
                             QDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QProgressDialog, QPushButton, QSizePolicy, QTableWidget,
                             QTableWidgetItem, QSlider, QSpinBox, QToolButton, QStyle,
                             QCheckBox, QGroupBox, QBoxLayout, QMessageBox, QAction,
                             QFileDialog, QMainWindow, QMessageBox, QTextEdit, QMenu, QFrame, )
from PyQt5.QtGui import (QDesktopServices, QKeySequence,QPainter,QPen)



import numpy as np
import time
import threading
import datetime

import cv2

from filelist import FileList
from imagefifo import FileImport

import config
import lineprofile as lp
import removebackground as rb
import noisefilter as nf



class MainWindow(QMainWindow):
    # Brightness.
    dirname = ''

    # step = 0

    def __init__(self):
        """ Constructor initializes a default value for the brightness, creates
            the main menu entries, and constructs a central widget that contains
            enough space for images to be displayed.

        """

        super(MainWindow, self).__init__()
       
        self.fileList = FileList()
        self.fileList.sig_file.connect(self.update_status)
        self.fileList.finished.connect(self.finish_process)
        self.makeDIcolor()

        self.createMenus()
        self.setCentralWidget(self.createCentralWidget())

        config.data_folder = config.get_savedparam("param", "initialdata_folder")

        
        
   
    def createMenus(self):
        """ Creates a main menu with two entries: a File menu, to allow the image
            to be selected, and a Brightness menu to allow the brightness of the
            separations to be changed.
            Initially, the Brightness menu items are disabled, but the first entry in
            the menu is checked to reflect the default brightness.
            メニューバーの作成
        """
        
        self.removebackground = QAction("&Remove Background", self)
        self.removebackground.setShortcut("Ctrl+B")
        self.removebackground.triggered.connect(self.MakeRemoveBackgroundWindow)
       

        self.noisefilter = QAction("&Noise Filter", self)
        self.noisefilter.setShortcut("Ctrl+N")
        self.noisefilter.triggered.connect(self.MakeNoiseFilterWindow)

        self.lineprofile = QAction("&Line", self)
        self.lineprofile.setShortcut("Ctrl+L")
        self.lineprofile.triggered.connect(self.MakeLineWindow)

        self.panelsetting = QAction("&Panel Setting", self)
        self.panelsetting.setShortcut("Ctrl+P")
        self.panelsetting.triggered.connect(self.getPanelSize)

        self.foldersetting = QAction("&Folder Setting", self)
        self.foldersetting.setShortcut("Ctrl+F")
        self.foldersetting.triggered.connect(self.initialfolder_setting)
      
        
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Image Processing')
        fileMenu.addAction(self.removebackground)
        fileMenu.addAction(self.noisefilter)

        fileMenu = menubar.addMenu('&Analysis')
        fileMenu.addAction(self.lineprofile)

        
        fileMenu = menubar.addMenu('&Setting')
        fileMenu.addAction(self.foldersetting)
        fileMenu.addAction(self.panelsetting)
        
    
    def createCentralWidget(self):
        """ Constructs a central widget for the window consisting of a two-by-two
            grid of labels, each of which will contain an image. We restrict the
            size of the labels to 256 pixels, and ensure that the window cannot
            be resized.
        """

        frame = QFrame(self)
        
        result = config.get_savedparam("panel", "FalconPy Main")
        if result is not None:
          # 一致する行が見つかった場合は、resultを処理する
           config.panel_left, config.panel_top, config.panel_width, config.panel_height = result
        else:
            config.panel_width= 300
            config.panel_height = 200
            config.panel_top = 100
            config.panel_left = 100
        
        self.resize(config.panel_width, config.panel_height)
        self.move(config.panel_left, config.panel_top)

        self.setWindowTitle('FalconPy Main')
        self.crFileControls("Select Files")
        self.crParamControls("Parameters")
        self.crCommentsControls("Comments")

        # self.paramsGroup.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.paramsGroup.setAlignment(Qt.AlignTop)
        grid = QHBoxLayout(frame)
        grid.addWidget(self.controlsGroup)
        grid.addWidget(self.paramsGroup)
        grid.addWidget(self.commentGroup)

        # layout = QHBoxLayout()
        # layout.addWidget(self.controlsGroup)
        # layout.addWidget(self.controlsGroup)
        # self.setLayout(layout)

        return frame
    
    def crFileControls(self, title):

        self.controlsGroup = QGroupBox(title)
        self.controlsGroup.setFixedWidth(330)

        self.txtFolder = QLineEdit()
        self.txtFolder.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.txtFolder.setReadOnly(1)
        # self.btnFolder = QPushButton("Open")
        # self.btnFolder.clicked.connect(self.show_folder_dialog)
        #self.filesFoundLabel = QLabel()

        iconSize = QSize(20, 20)
        self.btnFolder = QToolButton()
        self.btnFolder.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.btnFolder.setIconSize(iconSize)
        self.btnFolder.setToolTip("Open File")
        self.btnFolder.clicked.connect(self.show_folder_dialog)
        self.filesFoundLabel = QLabel()

        self.filesTable = QTableWidget(0, 2)
        self.filesTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.filesTable.setHorizontalHeaderLabels(("File Name", "Size"))
        self.filesTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        #self.filesTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.filesTable.resizeColumnToContents(0)
        self.filesTable.setStyleSheet("font-size: 11pt;")
        self.filesTable.verticalHeader().setDefaultSectionSize(5)
        self.filesTable.verticalHeader().hide()
        self.filesTable.setShowGrid(True)
        # self.filesTable.cellActivated.connect(self.openHeaderOfFile)
        self.filesTable.cellDoubleClicked.connect(self.openHeaderOfFile)
        self.filesTable.currentCellChanged.connect(self.changeOfFile)

        # self.filesTable.cellActivated.connect(self.openFileOfItem)
        self.txtLog = QTextEdit()
        self.txtLog.setReadOnly(True)

        self.pbar = QProgressBar()
        self.pbar.setTextVisible(False)

        cFrameLabel = QLabel("Current Frame:")
        self.frameSpinBox = QSpinBox()
        self.frameSpinBox.setRange(0, 100)
        self.frameSpinBox.setSingleStep(1)
        self.frameSpinBox.setEnabled(0)
        self.frameSpinBox.valueChanged.connect(self.SetF_SliderValue)

        self.dispmodeCheck = QCheckBox("Continuous")
        self.dispmodeCheck.toggle()
        self.dispmodeCheck.toggled.connect(self.CheckDispCont)

        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameSlider.setFocusPolicy(Qt.StrongFocus)
        self.frameSlider.setTickPosition(QSlider.TicksBothSides)
        self.frameSlider.setTickInterval(10)
        self.frameSlider.setSingleStep(1)
        self.frameSlider.setRange(0, 50)
        self.frameSlider.setEnabled(0)
        self.frameSlider.valueChanged.connect(self.SetF_SliderValue)

        #ここでsetf_slider valueを二回呼んでいる

        tFrameLabel = QLabel("Total frames:")
        # tFrameLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.tframeLineEdit = QLineEdit()
        self.tframeLineEdit.setText("")
        self.tframeLineEdit.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.tframeLineEdit.setReadOnly(1)
        # self.tframeLineEdit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        iconSize = QSize(20, 20)

        self.playButton = QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setEnabled(0)
        self.playButton.clicked.connect(self.movie_start)

        self.stopButton = QToolButton()
        self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stopButton.setIconSize(iconSize)
        self.stopButton.setToolTip("Stop")
        self.stopButton.setEnabled(0)
        self.stopButton.clicked.connect(self.movie_stop)

        # PlaybackLabel = QLabel("x play :")
        # self.playbackSpinBox = QSpinBox()
        # self.playbackSpinBox.setRange(1, 100)
        # self.playbackSpinBox.setSingleStep(1)
        # self.playbackSpinBox.setValue(config.pbSpeed)
        # self.playbackSpinBox.valueChanged.connect(self.Set_pbSpeed)

        self.pbar = QProgressBar()
        self.pbar.setTextVisible(False)

        hb1 = QHBoxLayout()

        hb1.addWidget(self.txtFolder)
        hb1.addWidget(self.btnFolder)

        # hb3 = QHBoxLayout()
        # hb3.addWidget(tFrameLabel)
        # hb3.addWidget(self.tframeLineEdit)

        #hb2 = QHBoxLayout()
        #hb2.addWidget(self.playButton)
        #hb2.addWidget(self.stopButton)
        #hb2.addWidget(PlaybackLabel)
        #hb2.addWidget(self.playbackSpinBox)

        layout = QGridLayout()
        
        layout.addLayout(hb1, 0, 0, 1, 4)
        
        layout.addWidget(self.filesFoundLabel, 1, 0, 1, 4)
        
        layout.addWidget(self.filesTable, 2, 0, 1, 4)
        
        layout.addWidget(cFrameLabel, 3, 0)
        layout.addWidget(self.frameSpinBox, 3, 1)
        layout.addWidget(self.dispmodeCheck, 3, 2)

        layout.addWidget(self.frameSlider, 4, 0, 1, 4)

        layout.addWidget(tFrameLabel, 5, 0)
        layout.addWidget(self.tframeLineEdit, 5, 1)

        #layout.addLayout(hb2, 6, 0, 1, 1)

        #layout.addWidget(self.pbar, 8, 0, 1, 4)
        
        self.controlsGroup.setLayout(layout)

    def crParamControls(self, title):

        self.paramsGroup = QGroupBox(title)

        self.DataTime = QLineEdit()
        self.DataTime.setText("  /  / ")
        self.DataTime.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.DataTime.setReadOnly(1)

        XpixelLabel = QLabel("X pixel:")
        self.Xpixel = QLineEdit()
        self.Xpixel.setText("")
        self.Xpixel.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.Xpixel.setReadOnly(1)

        YpixelLabel = QLabel("Y pixel:")
        self.Ypixel = QLineEdit()
        self.Ypixel.setText("")
        self.Ypixel.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.Ypixel.setReadOnly(1)

        XScansizeLabel = QLabel("X Scan Size:")
        self.XScansize = QLineEdit()
        self.XScansize.setText("")
        self.XScansize.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.XScansize.setReadOnly(1)

        YScansizeLabel = QLabel("Y Scan Size:")
        self.YScansize = QLineEdit()
        self.YScansize.setText("")
        self.YScansize.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.YScansize.setReadOnly(1)

        FrameTLabel = QLabel("Frame Time:")
        self.FrameT = QLineEdit()
        self.FrameT.setText("")
        self.FrameT.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.FrameT.setReadOnly(1)

        LaserLabel = QLabel("Laser:")
        self.Laser = QLineEdit()
        self.Laser.setText("")
        self.Laser.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.Laser.setReadOnly(1)

        OpeLabel = QLabel("Operator:")
        self.Ope = QLineEdit()
        self.Ope.setText("")
        self.Ope.setStyleSheet("QLineEdit { border: none; background-color: rgb(240,240,240) }")
        self.Ope.setReadOnly(1)

        layout = QGridLayout()

        layout.addWidget(self.DataTime, 0, 0, 1, 4)
        layout.addWidget(XpixelLabel, 1, 0)
        layout.addWidget(self.Xpixel, 1, 1)
        layout.addWidget(YpixelLabel, 2, 0)
        layout.addWidget(self.Ypixel, 2, 1)
        layout.addWidget(XScansizeLabel, 3, 0)
        layout.addWidget(self.XScansize, 3, 1)
        layout.addWidget(YScansizeLabel, 4, 0)
        layout.addWidget(self.YScansize, 4, 1)
        layout.addWidget(FrameTLabel, 5, 0)
        layout.addWidget(self.FrameT, 5, 1)
        layout.addWidget(LaserLabel, 6, 0)
        layout.addWidget(self.Laser, 6, 1)
        layout.addWidget(OpeLabel, 7, 0)
        layout.addWidget(self.Ope, 7, 1)

        self.paramsGroup.setLayout(layout)

    def crCommentsControls(self, title):

        self.commentGroup = QGroupBox(title)

        self.comment = QTextEdit()
        self.comment.setReadOnly(1)

        layout = QHBoxLayout()
        layout.addWidget(self.comment)
        layout.addStretch(20)

        self.commentGroup.setLayout(layout)
    
    def initialfolder_setting(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        dirname = QFileDialog.getExistingDirectory(self,
                                                   'open folder',
                                                   config.data_folder,
                                                   QFileDialog.ShowDirsOnly)
        
        if dirname is not None and dirname != "":
            config.data_folder = dirname 
        
            config.save_params("param", "initialdata_folder", config.data_folder)

    def show_folder_dialog(self):
        ''' open dialog and set to foldername '''
        
        dirname = QFileDialog.getExistingDirectory(self,
                                                   'open folder',
                                                  config.data_folder , #os.path.expanduser('.'),
                                                   QFileDialog.ShowDirsOnly)

        if dirname:
            self.dirname = dirname.replace('/', os.sep)  # ディレクトリの区切りをOSに合わせて変換しておく

            # "/"で分割し、最後の2つまでの要素だけを取得する
            folders = dirname.split("/")[-2:]
            # "/"で結合して、新しい文字列を作成する
            new_dirname = "/".join(folders)

            self.txtFolder.setText(new_dirname)
            # self.btnExec.setEnabled(True)
            self.step = 0

        self.file_open()


    def openHeaderOfFile(self, row, column):
        # item = self.filesTable.item(row, 0)
        # fv =  FileVariables
        config.row = row
        gh = FileImport(config.row)
        gh.getheader()

        config.index = 0
        image1ch=gh.OpenImage(config.files[config.row])
        self.UpdataFrame(0)
        # print("OKOK")
        self.showParams()

    def changeOfFile(self, currentrow, currentcolumn, previousrow, previousclumn):
        # item = self.filesTable.item(row, 0)
        if (currentrow != previousrow):
            config.row = currentrow
            config.index = 0

            gh = FileImport(config.row)
            gh.getheader()

            gh.OpenImage(config.files[config.row])

            self.frameSlider.setMaximum(config.FrameNum - 1)
            self.frameSpinBox.setMaximum(config.FrameNum - 1)
            self.tframeLineEdit.setText(str(config.FrameNum))

            self.UpdataFrame(0)

            self.showParams()

            self.CtlsActivation()

        
            
        if config.lineopen==True and config.lineclose==False:
           
           
            dl=lp.LineProfile()
            dl.MouseSet("img1ch")
            

    def file_open(self):
        # self.files = []
        if os.path.exists(self.dirname):
            # try:

            #print(self.dirname)

            QApplication.setOverrideCursor(Qt.WaitCursor)
            config.files = self.fileList.setup(self.dirname, '.asd')
            # print(self.files)
            # prnt("OK")
            # print(files)
            maxCnt = self.fileList.length
            if (maxCnt > 0):
                self.pbar.setValue(0)
                self.pbar.setMinimum(0)
                self.pbar.setMaximum(maxCnt)
                self.fileList.start()
                self.showFiles(config.files)
            else:
                QApplication.restoreOverrideCursor()
                QMessageBox.about(self, "Caution!", "asd files do not exist")

            QApplication.restoreOverrideCursor()
            # self.print_log('{0} is not exists'.format(self.dirname))

    # else:
    def showFiles(self, files):
        onlyfn = []
        for fn in files:
            # print(absolutePath(fn))
            # file = QFile(os.currentDir.absoluteFilePath(fn))
            file = QFile(os.path.abspath(fn))
            size = QFileInfo(file).size()
            onlyfn = QFileInfo(file).baseName()

            fileNameItem = QTableWidgetItem(onlyfn)
            fileNameItem.setFlags(fileNameItem.flags() ^ Qt.ItemIsEditable)
            sizeItem = QTableWidgetItem("%d KB" % (int((size + 1023) / 1024)))
            sizeItem.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            sizeItem.setFlags(sizeItem.flags() ^ Qt.ItemIsEditable)

            row = self.filesTable.rowCount()
            self.filesTable.insertRow(row)
            self.filesTable.setItem(row, 0, fileNameItem)
            self.filesTable.setItem(row, 1, sizeItem)

        config.FileNum = len(files)
        self.filesFoundLabel.setText("%d file(s) found (Click on a file to open it)" % len(files))

    def showParams(self):
        # layout.addWidget(self.DataTime,0,0)

        self.DataTime.setText(str(config.Year) + "/" + str(config.Month) + "/" + str(config.Day)
                              + "   " + str(config.Hour) + ":" + str(config.Minute) + ":" + str(config.Second))
        self.Xpixel.setText(str(config.XPixel) + " pixels")
        self.Ypixel.setText(str(config.YPixel) + " pixels")
        self.XScansize.setText(str(config.XScanSize) + " nm")
        self.YScansize.setText(str(config.YScanSize) + " nm")
        self.FrameT.setText(str(config.FrameTime) + " ms")
        if (config.LaserFlag == 1):
            self.Laser.setText("ON")
        else:
            self.Laser.setText("OFF")
        self.Ope.setText(config.OpeName)

        self.comment.setText(config.Comment)

   
    def getPanelSize(self):
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QWidget):
                 title = widget.windowTitle()
                 config.panel_height = widget.geometry().height()
                 config.panel_width= widget.geometry().width()
                 config.panel_left = widget.geometry().left()
                 config.panel_top = widget.geometry().top()

                 
                 config.save_params("panel",title, 0)               
 
            # result = self.get_savedparam("panel", "FalconPy Main")

            # if result is not None:
            #     # 一致する行が見つかった場合は、resultを処理する
            #     val2, val3, val4, val5 = result
            #     print(f"2列目の値: {val2}")
            #     print(f"3列目の値: {val3}")
            #     print(f"4列目の値: {val4}")
            #     print(f"5列目の値: {val5}")
            # else:
            #     pass
            
            # with open("falconviewer.parm", "w") as file:
            #     # ファイルの最初の行に"FalconPy Main", 0, 1, 2, 3を書き込む
            #     file.write("FalconPy Main, 0, 1, 2, 3\n")
    
            #     # 2行目に"Test", 1, 2, 3, 4を書き込む
            #     file.write("Test, 1, 2, 3, 4\n")

            #ファイルを読み込んで一行ずつ処理する
            # with open("Falconviewer.parm", "r") as file:
            #     lines = file.readlines()

            # # 行ごとに処理する
            # for i, line in enumerate(lines):
    
            #     # 行に"Test"が含まれている場合
            #     if "Test" in line:
        
            #         # 行をカンマで分割し、3番目の数値を100に変更する
            #         parts = line.strip().split(",")
            #         parts[2] = "30"
        
            #         # 変更後の行を作成する
            #         new_line = ",".join(parts) + "\n"
        
            #         # 変更後の行に置き換える
            #         lines[i] = new_line

            # # ファイルを書き込みモードで開いて、変更後の内容を書き込む
            # with open("Falconviewer.parm", "w") as file:
            #     file.writelines(lines)



    

    


    def SetF_SliderValue(self,value):
        # self.frameSlider.setValue(value)

        config.index = value

        gh = FileImport(config.row)
        gh.OpenImage(config.files[config.row])


        self.UpdataFrame(value)        

        if config.lineopen==True and config.lineclose==False:
           
            dl=lp.LineProfile()
            dl.MouseSet("img1ch")
           

    def Set_pbSpeed(self, value):
        # self.frameSlider.setValue(value)
        config.pbSpeed = value

        # self.frameSpinBox.setValue(config.index)

   

    def keyPressEvent(self, e):
        gh = FileImport(config.row)
        udFrame = 0
        if e.key() == Qt.Key_Left:
            config.index -= 1
            udFrame = config.index
            if (config.index >= 0):
                gh.OpenImage(config.files[config.row])
            else:
                if (config.DispMode == True and config.row > 0):
                    config.row -= 1
                    gh = FileImport(config.row)
                    gh.getheader()
                    udFrame = config.FrameNum - 1
                    config.index = config.FrameNum - 1
                    gh.OpenImage(config.files[config.row])
                    self.filesTable.setCurrentCell(config.row, 1)
                    self.frameSlider.setMaximum(udFrame)
                    self.frameSpinBox.setMaximum(udFrame)
                    # self.frameSlider.setValue(testval)
                    # self.frameSpinBox.setValue(testval)

                else:
                    config.index = 0
                    udFrame = config.index

        elif e.key() == Qt.Key_Right:
            config.index += 1
            udFrame = config.index
            if (config.index <= config.FrameNum - 1):
                gh.OpenImage(config.files[config.row])
            else:
                if (config.DispMode == True and config.row < config.FileNum):
                    config.index = 0
                    config.row += 1
                    udFrame = config.index
                    gh = FileImport(config.row)
                    gh.getheader()
                    gh.OpenImage(config.files[config.row])

                    self.filesTable.setCurrentCell(config.row, 1)
                    # self.frameSlider.setValue(config.index)
                    # self.frameSpinBox.setValue(config.index)
                    self.frameSlider.setMaximum(config.FrameNum - 1)
                    self.frameSpinBox.setMaximum(config.FrameNum - 1)

                else:
                    config.index = config.FrameNum - 1
                    # testval = config.index
        # self.frameSlider.setValue(config.index)
        # self.frameSpinBox.setValue(config.index)
        # udFrame = config.index
        self.UpdataFrame(udFrame)

        # elf.slider.setMaximum(config.FrameNum)

        #    self.close()

    def UpdataFrame(self, value):
        self.frameSpinBox.setValue(value)
        self.frameSlider.setValue(value)
        self.filesTable.setCurrentCell(config.row, 1)

    
    def CheckDispCont(self, state):
        config.DispMode = state

    def CtlsActivation(self):
        self.frameSpinBox.setEnabled(1)
        self.frameSlider.setEnabled(1)
        self.playButton.setEnabled(1)

    

    def movie_start(self):
        # self.frUpdata = FrameUpData()
        self.playButton.setEnabled(0)
        self.stopButton.setEnabled(1)
        self.frUpdata = FrameUpData()
        self.frUpdata.sig_frame.connect(self.frupdate_status)
        # self.frUpdata.finished.connect(self.frupdate_finishprocess)

        config.movie_flag = 1

        self.frUpdata.start()

    def movie_stop(self):
        self.frUpdata = FrameUpData()
        config.movie_flag = 0

        self.playButton.setEnabled(1)
        self.stopButton.setEnabled(0)

        # self.finished.emit()

    def frupdate_status(self, framenum):
        # print(framenum)
        self.frameSpinBox.setValue(framenum)
        self.frameSlider.setValue(framenum)
        self.filesTable.setCurrentCell(config.row, 1)

    # def frupdate_finishprocess(self):
    #    pass

    

    def makeDIcolor(self):

        for i in range(0, 255):

            if (i < 158):
                config.DIcolor[i, 0, 2] = i / 157 * 255
            elif (i >= 158):
                config.DIcolor[i, 0, 2] = 255

            config.DIcolor[i, 0, 1] = i

            if (i < 176):
                config.DIcolor[i, 0, 0] = 0
            elif (i >= 176):
                config.DIcolor[i, 0, 0] = 67 + 188 * (i - 176) / 79

    def update_status(self, filename):
        self.txtLog.append(filename)
        self.step += 1
        self.pbar.setValue(self.step)  # progressBarを進める

    def finish_process(self):
        self.fileList.wait()
        # 実行ボタンを隠す
        # self.btnExec.setEnabled(False)
        # self.btnExec.setVisible(False)
        # 終了ボタンを表示する
        # self.btnExit.setEnabled(True)
        # self.btnExit.setVisible(True)
    
    
    def MakeLineWindow(self):
        self.linewindow=lp.LineWindow()
        config.lineopen=True
        config.lineclose=False
        if config.lineopen==True and config.lineclose==False:
            dl=lp.LineProfile()
            dl.MakeProfileWindow()
            dl.MouseSet("img1ch")
    
    def MakeRemoveBackgroundWindow(self):
        self.Removebackgroundwindow=rb.RemovebackgroundWindow()
        self.Removebackgroundwindow.show()


    def MakeNoiseFilterWindow(self):
        self.Noisefilterwindow=nf.NoisefilterWindow()
        self.Noisefilterwindow.show()
    
    #def  MakeSettingWindow(self):
    #    self.Foldersettingwindow=st.FoldersettingWindow()
    #    self.Foldersettingwindow.show()

       

class FrameUpData(QThread):
    """docstring for TestThread"""
    sig_frame = pyqtSignal(int)

    def __init__(self, parent=None):
        super(FrameUpData, self).__init__(parent)

        self.stopped = False
        self.mutex = QMutex()

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True
        # print("Finished")

    def run(self):
        self.stopped = False
        # print("Here")
        m = MainWindow()

        while (config.movie_flag == 1):
            print(config.pbSpeed)
            time.sleep(config.FrameTime / 1000 / config.pbSpeed)
            config.index += 1
            gh = FileImport(config.row)

            if (config.index <= config.FrameNum - 1):
                gh.OpenImage(config.files[config.row])
            else:
                if (config.DispMode == True and config.row < config.FileNum):
                    config.index = 0
                    config.row += 1
                    # print(config.row)
                    gh.getheader()
                    gh.OpenImage(config.files[config.row])

                    # self.frameSlider.setValue(config.index)
                    # self.frameSpinBox.setValue(config.index)
                    # self.frameSlider.setMaximum(config.FrameNum-1)
                    # self.frameSpinBox.setMaximum(config.FrameNum-1)

                else:
                    config.index = config.FrameNum - 1
                    config.movie_flag = 0

            udFrame = config.index
            self.sig_frame.emit(udFrame)

        if (config.movie_flag == 0):
            self.stop()
            self.finished.emit  # signal送信

        # m.frameSpinBox.setValue(config.index)



def main():
    pass

 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()

    main_window.show()
    sys.exit(app.exec_())
