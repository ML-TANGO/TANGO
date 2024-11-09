# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import cv2
import GuiUtils
import Utils
import numpy as np
# import processAi

class SingleImage(QLabel):
    DRAG_ACTION_NONE = -1
    DRAG_ACTION_ITEM_CREATE = 0
    DRAG_ACTION_ITEM_RESIZE = 1
    DRAG_ACTION_ITEM_MOVE = 2
    DRAG_ACTION_FRAME_MOVE_XY = 3
    DRAG_ACTION_FRAME_MOVE_X = 4
    DRAG_ACTION_FRAME_MOVE_Y = 5

    FRAME_DRAW_MODE_FIT = 0
    FRAME_DRAW_MODE_ZOOM = 1

    def __init__(self, parent):
        QLabel.__init__(self, parent)
        #print("BBoxImage.__init__()")
        self.setAcceptDrops(True)
        self.imageFile = None
        self.pixmap = None
        self.currentData = None

        self.detectionResults = None

        self.frameDrawMode = SingleImage.FRAME_DRAW_MODE_ZOOM
        self.dragAction = SingleImage.DRAG_ACTION_NONE
        self.zoomScale = 1.0
        self.zoomedWidth = 0
        self.zoomedHeight = 0
        self.centerX = 0
        self.centerY = 0
        self.leftAtZoom = 0
        self.topAtZoom = 0
        self.mouseX = 0
        self.mouseY = 0
        self.CX = 0
        self.CY = 0
        self.mX1 = 0
        self.mY1 = 0
        self.mX2 = 0
        self.mY2 = 0

        self.frameFont = QFont()
        self.frameFont.setFamily("Arial")
        self.frameFont.setPointSize(20)
        self.setFont(self.frameFont)

        self.pixmapTransform = False
        self.showUndistoredState = False

    def clearImage(self):
        self.imageFile = None
        self.pixmap = None
        self.repaint()

    def setImageFromFile(self, imageFile):
        self.imageFile = imageFile
        self.pixmap = QPixmap(self.imageFile)
        self.initPosition()
        self.repaint()

    def setToFitView(self):
        self.frameDrawMode = SingleImage.FRAME_DRAW_MODE_FIT
        self.repaint()

    def setToZoomView(self):
        self.frameDrawMode = SingleImage.FRAME_DRAW_MODE_ZOOM
        self.repaint()

    def isFitViewMode(self):
        return self.frameDrawMode == SingleImage.FRAME_DRAW_MODE_FIT

    def mouseMoveEvent(self, ev):
        button = ev.buttons()
        pos = ev.pos()
        self.mouseX = pos.x()
        self.mouseY = pos.y()
        if button == Qt.LeftButton: # and ev.modifiers() == Qt.ControlModifier:
            if self.dragAction == SingleImage.DRAG_ACTION_FRAME_MOVE_XY:
                deltaX = self.CX - pos.x() 
                deltaY = self.CY - pos.y()
                self.leftAtZoom -= deltaX
                self.topAtZoom -= deltaY
                self.CX = pos.x()
                self.CY = pos.y()
                if self.leftAtZoom > 0:
                    self.leftAtZoom = 0
                else:
                    if self.leftAtZoom + self.zoomedWidth < self.width():
                        self.leftAtZoom = self.width() - self.zoomedWidth
                if self.topAtZoom > 0:
                    self.topAtZoom = 0
                else:
                    if self.topAtZoom + self.zoomedHeight < self.height():
                        self.topAtZoom = self.height() - self.zoomedHeight
                self.repaint()
                print("leftAtZoom = {}".format(self.leftAtZoom))
                print("topAtZoom = {}".format(self.topAtZoom))
                print("zoomedWidth = {}".format(self.zoomedWidth))
                print("zoomedHeight = {}".format(self.zoomedHeight))
            elif self.dragAction == SingleImage.DRAG_ACTION_FRAME_MOVE_X:
                deltaX = self.CX - pos.x() 
                self.leftAtZoom -= deltaX
                self.CX = pos.x()
                if self.leftAtZoom > 0:
                    self.leftAtZoom = 0
                else:
                    if self.leftAtZoom + self.zoomedWidth < self.width():
                        self.leftAtZoom = self.width() - self.zoomedWidth
                self.repaint()
            elif self.dragAction == SingleImage.DRAG_ACTION_FRAME_MOVE_Y:
                deltaY = self.CY - pos.y()
                self.topAtZoom -= deltaY
                self.CY = pos.y()
                if self.topAtZoom > 0:
                    self.topAtZoom = 0
                else:
                    if self.topAtZoom + self.zoomedHeight < self.height():
                        self.topAtZoom = self.height() - self.zoomedHeight
                self.repaint()
    
    def mouseReleaseEvent(self, ev):
        # print("Release:", x, y)
        self.repaint()

    def mousePressEvent(self, ev):
        button = ev.buttons()
        pos = ev.pos()

        if button == Qt.LeftButton:        # item Add
            self.CX = pos.x()
            self.CY = pos.y()

            if self.zoomedWidth > self.width() and self.zoomedHeight <= self.height():
                self.dragAction = SingleImage.DRAG_ACTION_FRAME_MOVE_X
            elif self.zoomedWidth <= self.width() and self.zoomedHeight > self.height():
                self.dragAction = SingleImage.DRAG_ACTION_FRAME_MOVE_Y
            elif self.zoomedWidth > self.width() and self.zoomedHeight > self.height():
                self.dragAction = SingleImage.DRAG_ACTION_FRAME_MOVE_XY
            else:
                self.dragAction = SingleImage.DRAG_ACTION_NONE
            return
            # else:
            #     # print("Press: channel", self.chIndex, ev.pos().x(), ev.pos().y())
            #     if self.mainWindow.getSelectedChannel() != self.chIndex:
            #         self.mainWindow.setSelectedChannel(self.chIndex)
            #         self.repaint()

    def zoomIn(self):
        self.zoomScale += 0.2
        if self.zoomScale > 3.0:
            self.zoomScale = 3.0
        self.rescaleImageAfterZooming()
        self.repaint()
        # self.postProcessAfterZooming()

    def zoomOut(self):
        self.zoomScale -= 0.2
        if self.zoomScale < 0.2:
            self.zoomScale = 0.2
        self.rescaleImageAfterZooming()
        self.repaint()
        # self.postProcessAfterZooming()

    def rescaleImageAfterZooming(self):
        self.zoomedWidth = int(self.sr.right()*self.zoomScale)
        self.zoomedHeight = int(self.sr.bottom()*self.zoomScale)
        self.scx = int(self.width()/2)       # screen center X
        self.scy = int(self.height()/2)      # screen center Y
        self.leftAtZoom = self.scx - int(self.zoomedWidth/2)
        self.topAtZoom = self.scy - int(self.zoomedHeight/2)

    def postProcessAfterZooming(self):
        # self.leftAtZoom -= self.dx*self.zoomScale
        # self.topAtZoom -= self.dy*self.zoomScale
        # return
        redrawNeeded = False
        if self.zoomedWidth < self.width():
            self.leftAtZoom = 0
            redrawNeeded = True
        else:
            if self.leftAtZoom + self.zoomedWidth < self.width():
                self.leftAtZoom = self.width() - self.zoomedWidth
                redrawNeeded = True
        if self.zoomedHeight < self.height():
            self.topAtZoom = 0
            redrawNeeded = True
        else:
            if self.topAtZoom + self.zoomedHeight < self.height():
                self.topAtZoom = self.height() - self.zoomedHeight
                redrawNeeded = True
        if redrawNeeded == True:
            self.repaint()

    def wheelEvent(self, event):
        # if event.modifiers() == Qt.ControlModifier:
        delta = event.angleDelta()
        pos = event.pos()
        # self.wx = pos.x()
        # self.wy = pos.y()
        # print(self.wx, self.wy)
        # self.dx = self.scx - self.wx
        # self.dy = self.scy - self.wy
        if delta.y() > 0:
            self.zoomIn()
        else:
            self.zoomOut()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.doPaint(event, qp)
        qp.end()

    def initPosition(self):
        if self.pixmap == None:
            return
        iw = self.pixmap.width()
        ih = self.pixmap.height()
        self.sr = QRect(0, 0, iw, ih)            # source rect of image
        self.zoomedWidth = int(self.sr.right()*self.zoomScale)
        self.zoomedHeight = int(self.sr.bottom()*self.zoomScale)
        self.scx = int(self.width()/2)       # screen center X
        self.scy = int(self.height()/2)      # screen center Y
        self.leftAtZoom = self.scx - int(self.zoomedWidth/2)
        self.topAtZoom = self.scy - int(self.zoomedHeight/2)
        # self.setCenter(int(self.zoomedWidth/2), int(self.zoomedWidth/2))

    def setCenter(self, cx, cy):
        self.centerX = cx
        self.centerY = cy

    def resizeEvent(self, ev):
        self.initPosition()
        self.repaint()
        
    def doPaint(self, event, qp):
        w = self.width()
        h = self.height()

        if self.pixmap == None:
            return
        iw = 0
        iw = self.pixmap.width()
        if iw == 0:
            Utils.logPrint("SingleImage : doPaint() - image width is ZERO. Return...")
            return
        ih = self.pixmap.height()
        hpixmap = QPixmap(iw, ih)
        painter = QPainter(hpixmap)
        px = 0
        painter.drawPixmap(px, 0, self.pixmap)
        # px += self.pixmaps[i].width()
        painter.end()
        if self.frameDrawMode == SingleImage.FRAME_DRAW_MODE_FIT:
            if self.pixmapTransform:
                scaledPixmap = hpixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                scaledPixmap = hpixmap.scaled(w, h, Qt.KeepAspectRatio)
            pw = scaledPixmap.width()
            ph = scaledPixmap.height()
            qp.drawPixmap(int((w-pw)/2), int((h-ph)/2), scaledPixmap)
            xx = int((w-pw)/2)
            yy = int((h-ph)/2)
            ww = int(scaledPixmap.width()/3)
            hh = int(scaledPixmap.height())

            self.imageRect = []
            self.imageRect.append(QRect(xx, yy, ww, hh))
            self.imageRect.append(QRect(xx+ww, yy, ww, hh))
            self.imageRect.append(QRect(xx+ww*2, yy, ww, hh))
            if self.detectionResults != None:
                self.drawDetectionResultsFit(qp, self.detectionResults, iw, ih, int((w-pw)/2), int((h-ph)/2), pw, ph)
        else:
            if self.pixmapTransform:
                qp.setRenderHint(QPainter.SmoothPixmapTransform)
            self.tr = QRect(self.leftAtZoom, self.topAtZoom, self.zoomedWidth, self.zoomedHeight)
            qp.drawPixmap(self.tr, hpixmap, self.sr)
            # sr = QRect(0, 0, iw, ih)
            # self.zoomedWidth = int(sr.right()*self.zoomScale)
            # # Utils.logPrint("Zoomed Width = {}".format(self.zoomedWidth))
            # self.zoomedHeight = int(sr.bottom()*self.zoomScale)
            # self.tr = QRect(self.leftAtZoom, self.topAtZoom, self.zoomedWidth, self.zoomedHeight)
            # # Utils.logPrint("TARGET_RECT = {}, {}, {}, {}".format(self.leftAtZoom, self.topAtZoom, self.zoomedWidth, self.zoomedHeight))
            # if self.pixmapTransform:
            #     qp.setRenderHint(QPainter.SmoothPixmapTransform)
            # qp.drawPixmap(self.tr, hpixmap, sr)
            
            # xx = self.leftAtZoom
            # yy = self.topAtZoom
            # ww = int(self.zoomedWidth/3)
            # hh = int(self.zoomedHeight)
            # self.imageRect = []
            # self.imageRect.append(QRect(xx, yy, ww, hh))
            # self.imageRect.append(QRect(xx+ww, yy, ww, hh))
            # self.imageRect.append(QRect(xx+ww*2, yy, ww, hh))

            # if self.detectionResults != None:
            #     self.drawDetectionResultsZoom(qp, self.detectionResults, self.tr, sr)


    def drawDetectionResultsZoom(self, qp, dr, tr, sr):
        fx = float(tr.width()) / float(sr.width())
        fy = float(tr.height()) / float(sr.height())
        xs = [0, int(tr.width()/3), int(tr.width()*2/3)]
        y = 0
        # for i in range(self.channels):
        for i in range(len(dr["Results"])):
            result = dr["Results"][i]
            idx = dr["Results"][i]["cam"]
            x = xs[idx-1]
            for obj in result["Objects"]:
                c = obj["class"]
                l = obj["bbox"]["left"]
                t = obj["bbox"]["top"]
                r = obj["bbox"]["right"]
                b = obj["bbox"]["bottom"]
                if obj["confidence"] > self.appSettings["Events"]["FireDetection"]["MinimumConfidence"] and obj["class"] >= 7:
                    d_l = int(fx*l) + x + tr.left()
                    d_t = int(fy*t) + y + tr.top()
                    d_r = int(fx*r) + x + tr.left()
                    d_b = int(fy*b) + y + tr.top()
                    d_w = d_r - d_l + 1
                    d_h = d_b - d_t + 1
                    lineColor = QColor(self.appSettings["objectDetection"]["boxLineColor"][c][0],
                            self.appSettings["objectDetection"]["boxLineColor"][c][1],
                            self.appSettings["objectDetection"]["boxLineColor"][c][2])
                    self.drawBoxWithPicker(qp, d_l, d_t, d_r, d_b, lineColor, 3, False)
                    self.drawBoxWithPicker(qp, d_l, d_t, d_r, d_b, QColor(255, 255, 255, 255), 1, False)
                    resultText = "%s(%d%%)" % (self.appSettings["objectDetection"]["label"][obj["className"]], int(obj["confidence"]*100))
                    self.drawOutlinedText(qp, d_l+3, d_t+3, resultText, QColor(255, 255, 255), QColor(0, 128, 0))
                    # qp.drawRect(d_l, d_t, d_w, d_h)
                    # qp.drawText(d_l, d_t, obj["className"])
        return
    
    def drawDetectionResultsFit(self, qp, dr, iw, ih, sx, sy, w, h):
        fx = float(w) / float(iw)
        fy = float(h) / float(ih)
        xs = [0, int(w/3), int(w*2/3)]
        # for i in range(self.channels):
        for i in range(len(dr["Results"])):
            result = dr["Results"][i]
            idx = dr["Results"][i]["cam"]
            x = xs[idx-1]
            for obj in result["Objects"]:
                c = obj["class"]
                l = obj["bbox"]["left"]
                t = obj["bbox"]["top"]
                r = obj["bbox"]["right"]
                b = obj["bbox"]["bottom"]
                if obj["confidence"] > self.appSettings["Events"]["FireDetection"]["MinimumConfidence"] and obj["class"] >= 7:
                    d_l = int(fx*l) + x + sx
                    d_t = int(fy*t) + sy
                    d_r = int(fx*r) + x + sx
                    d_b = int(fy*b) + sy
                    d_w = d_r - d_l + 1
                    d_h = d_b - d_t + 1
                    lineColor = QColor(self.appSettings["objectDetection"]["boxLineColor"][c][0],
                            self.appSettings["objectDetection"]["boxLineColor"][c][1],
                            self.appSettings["objectDetection"]["boxLineColor"][c][2])
                    self.drawBoxWithPicker(qp, d_l, d_t, d_r, d_b, lineColor, 3, False)
                    self.drawBoxWithPicker(qp, d_l, d_t, d_r, d_b, QColor(255, 255, 255, 255), 1, False)
                    resultText = "%s(%d%%)" % (self.appSettings["objectDetection"]["label"][obj["className"]], int(obj["confidence"]*100))
                    self.drawOutlinedText(qp, d_l+3, d_t+3, resultText, QColor(255, 255, 255), QColor(0, 128, 0))

    def drawBoxWithPicker(self, qp, left, top, right, bottom, pc, pw, drawPicker):
        # left, right = Utils.swap(left, right)
        # top, bottom = Utils.swap(top, bottom)
        width = right - left + 1
        height = bottom - top + 1
        rect = QRect(left, top, width, height)
        pen = qp.pen()
        pen.setWidth(pw)
        pen.setColor(pc)
        qp.setPen(pen)  #QColor(255, 0, 0))
        qp.setBrush(Qt.NoBrush)
        qp.drawRect(rect)

        if drawPicker == True:
            qp.setBrush(pc)
            pen.setWidth(0)
            center = QPoint(left, top)
            qp.drawEllipse(center, 4, 4)
            center = QPoint(left+int(width/2), top)
            qp.drawEllipse(center, 4, 4)
            center = QPoint(right+1, top)
            qp.drawEllipse(center, 4, 4)

            center = QPoint(left, top+int(height/2))
            qp.drawEllipse(center, 4, 4)
            center = QPoint(right+1, top+int(height/2))
            qp.drawEllipse(center, 4, 4)
            
            center = QPoint(left, bottom+1)
            qp.drawEllipse(center, 4, 4)
            center = QPoint(left+int(width/2), bottom+1)
            qp.drawEllipse(center, 4, 4)
            center = QPoint(right, bottom+1)
            qp.drawEllipse(center, 4, 4)

            center = QPoint(left+int(width/2), top+int(height/2))
            qp.drawEllipse(center, 4, 4)

    def drawOutlinedText(self, qp, l, t, text, tc, oc):
        width = self.fontMetrics().boundingRect(text).width()
        height = self.fontMetrics().boundingRect(text).height()
        rect = QRect(l, t-1, width, height)
        qp.setPen(oc)
        qp.drawText(rect, Qt.AlignLeft | Qt.AlignTop, text)
        rect = QRect(l-1, t, width, height)
        qp.drawText(rect, Qt.AlignLeft | Qt.AlignTop, text)
        rect = QRect(l+1, t, width, height)
        qp.drawText(rect, Qt.AlignLeft | Qt.AlignTop, text)
        rect = QRect(l, t+1, width, height)
        qp.drawText(rect, Qt.AlignLeft | Qt.AlignTop, text)

        rect = QRect(l, t, width, height)
        qp.setPen(tc)
        qp.drawText(rect, Qt.AlignLeft | Qt.AlignTop, text)
