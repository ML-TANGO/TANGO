# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

def getSelectedIndexOnTreeView(tv: QTreeView):
    index = tv.selectedIndexes()[0]
    return index

def openFileDialog(win, title, path, exts):
    ''' Function  openFileDialog'''
    fileNames = QFileDialog.getOpenFileNames(win, title, path, exts)[0]
    if len(fileNames) > 0:
        return len(fileNames), fileNames
    return 0, None

def saveFileDialog(win, title, path, exts):
    ''' Function  saveFileDialog'''
    fileName = QFileDialog.getSaveFileName(win, title, path, exts)
    if fileName:
        return fileName
    return None

def getFileDirectory(win, title):
    folderDir = QFileDialog.getExistingDirectory(win, title)
    if folderDir:
        return folderDir
    return None