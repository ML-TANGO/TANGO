# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class SingleImage(QLabel):
    FRAME_DRAW_MODE_FIT = 0
    FRAME_DRAW_MODE_ZOOM = 1

    def __init__(self, parent):
        QLabel.__init__(self, parent)
        self.setAcceptDrops(True)
        self.imageFile = None
        self.pixmap = None
        self.currentData = None

        self.detectionResults = None

        self.frameDrawMode = SingleImage.FRAME_DRAW_MODE_ZOOM

        self.drag = False
        self.lastX = 0
        self.lastY = 0
        self.imageX = -1
        self.imageY = -1
        self.dispOffsetX = 0
        self.dispOffsetY = 0
        self.dispZoom = 1.0

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
        if self.drag == False:
            return
        button = ev.buttons()
        pos = ev.pos()
        ptx = pos.x()
        pty = pos.y()
        xoff, yoff, scale, wBmp, hBmp = self.calcDrawParam()

        self.imageX = int((ptx - xoff) / scale)
        self.imageY = int((pty - yoff) / scale)

        dx = ptx - self.lastX
        dy = pty - self.lastY
        self.dispOffsetX += dx
        self.dispOffsetY += dy
        self.lastX = ptx
        self.lastY = pty
        self.repaint()

    def mouseReleaseEvent(self, ev):
        # print("Release:", x, y)
        self.drag = False
        self.repaint()

    def mousePressEvent(self, ev):
        if self.pixmap == None:
            return
        button = ev.buttons()

        if button == Qt.LeftButton:  # item Add
            self.drag = True
            pos = ev.pos()
            self.lastX = pos.x()
            self.lastY = pos.y()

    def wheelEvent(self, event):
        delta = event.angleDelta()
        pos = event.pos()
        if delta.y() > 0:
            scale = 1.1
        else:
            scale = 1.0 / 1.1

        if self.dispZoom * scale <= 0.1:
            return
        if self.dispZoom * scale >= 10.0:
            return

        wView = self.width()
        hView = self.height()

        cxImage = wView / 2 + self.dispOffsetX
        cyImage = hView / 2 + self.dispOffsetY

        dx = cxImage - pos.x()
        dy = cyImage - pos.y()

        self.dispOffsetX += int(dx * scale - dx)
        self.dispOffsetY += int(dy * scale - dy)
        self.dispZoom *= scale
        self.repaint()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.doPaint(event, qp)
        qp.end()

    def initPosition(self):
        if self.pixmap == None:
            return
        self.imageX = -1
        self.imageY = -1
        self.dispOffsetX = 0
        self.dispOffsetY = 0
        self.dispZoom = 1.0
        self.lastX = 0
        self.lastY = 0

    def resizeEvent(self, ev):
        self.initPosition()
        self.repaint()

    def calcDrawParam(self):
        x = 0
        y = 0
        w = self.width()
        h = self.height()
        sx = self.pixmap.width()
        sy = self.pixmap.height()
        hDraw = h
        wDraw = (sx * hDraw) / sy
        if wDraw > w:
            wDraw = w
            hDraw = (wDraw * sy) / sx

        x += (w - wDraw) / 2
        y += (h - hDraw) / 2

        x += self.dispOffsetX
        y += self.dispOffsetY

        wBmp = int(wDraw * self.dispZoom)
        hBmp = int(hDraw * self.dispZoom)
        xoff = x - (wBmp - wDraw) / 2
        yoff = y - (hBmp - hDraw) / 2

        xscale = wDraw / sx * self.dispZoom
        yscale = hDraw / sy * self.dispZoom
        scale = min(xscale, yscale)

        return xoff, yoff, scale, wBmp, hBmp

    def doPaint(self, event, qp):
        if self.pixmap == None:
            return
        if self.frameDrawMode == SingleImage.FRAME_DRAW_MODE_FIT:
            self.drawFitImage(qp, self.pixmap)
        else:
            self.drawZoomedImage(qp, self.pixmap)

    def drawFitImage(self, qp, pixmap):
        w = self.width()
        h = self.height()
        iw = self.pixmap.width()
        ih = self.pixmap.height()
        if self.pixmapTransform:
            scaledPixmap = pixmap.scaled(
                w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            scaledPixmap = pixmap.scaled(w, h, Qt.KeepAspectRatio)
        pw = scaledPixmap.width()
        ph = scaledPixmap.height()
        qp.drawPixmap(int((w - pw) / 2), int((h - ph) / 2), scaledPixmap)
        xx = int((w - pw) / 2)
        yy = int((h - ph) / 2)
        ww = int(scaledPixmap.width() / 3)
        hh = int(scaledPixmap.height())

        self.imageRect = []
        self.imageRect.append(QRect(xx, yy, ww, hh))
        self.imageRect.append(QRect(xx + ww, yy, ww, hh))
        self.imageRect.append(QRect(xx + ww * 2, yy, ww, hh))
        if self.detectionResults != None:
            self.drawDetectionResultsFit(
                qp,
                self.detectionResults,
                iw,
                ih,
                int((w - pw) / 2),
                int((h - ph) / 2),
                pw,
                ph,
            )

    def drawZoomedImage(self, qp, pixmap):
        xoff, yoff, scale, wBmp, hBmp = self.calcDrawParam()
        print(xoff, yoff, scale, wBmp, hBmp)
        self.tr = QRect(int(xoff), int(yoff), int(wBmp), int(hBmp))
        self.sr = QRect(0, 0, self.pixmap.width(), self.pixmap.height())
        qp.drawPixmap(self.tr, pixmap, self.sr)
