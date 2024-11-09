from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class ThumbnailListWidgetItem(QListWidgetItem):
    def __init__(self, parent=None):
        QListWidgetItem.__init__(self, parent)
        self.imageName = None
        self.imagePath = None

    def setImageInfo(self, imageName, imagePath):
        self.imageName = imageName
        self.imagePath = imagePath
        # print(self.getImageInfo())

    def getImageInfo(self):
        return self.imageName, self.imagePath


class ThumbnailListWidget(QWidget):
    def __init__(self, parent=None):
        super(ThumbnailListWidget, self).__init__(parent)
        # Icon Mode
        self.allQVBoxLayout = QVBoxLayout()
        self.iconQLabel = QLabel()
        self.allQVBoxLayout.addWidget(self.iconQLabel, 0)
        self.textDownQLabel = QLabel()
        self.textDownQLabel.setAlignment(Qt.AlignHCenter)
        self.allQVBoxLayout.addWidget(self.textDownQLabel, 1)
        self.setLayout(self.allQVBoxLayout)
        self.controlPressed = False

        font = QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(11)
        self.textDownQLabel.setFont(font)

        # setStyleSheet
        self.textDownQLabel.setStyleSheet(
            """
            color: rgb(0, 0, 0);
        """
        )

    def makeShortName(self, imageName):
        l = len(imageName)
        if l > 10:
            return "{} ... {}".format(imageName[0:6], imageName[l - 2 : l])
        return imageName

    def setTextDown(self, text):
        self.textDownQLabel.setText(text)

    def setIcon(self, imagePath):
        basePixmap = QPixmap(100, 100)

        if isinstance(imagePath, str):
            pixmap = QPixmap(imagePath)
        elif isinstance(imagePath, QPixmap):
            pixmap = imagePath
        self.iconQLabel.setPixmap(pixmap)
        pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio)
        w = pixmap.size().width()
        h = pixmap.size().height()
        # print(w)
        # print(h)
        painter = QPainter()
        painter.begin(basePixmap)
        color = QColor(255, 255, 255)
        painter.fillRect(0, 0, 100, 100, color)
        painter.drawPixmap(int((100 - w) / 2), int((100 - h) / 2), pixmap)
        painter.end()
        self.iconQLabel.setPixmap(basePixmap)

    def isControlPressed(self):
        return self.controlPressed

    def mousePressEvent(self, ev):
        button = ev.button()
        if button == Qt.LeftButton and ev.modifiers() == Qt.ControlModifier:
            self.controlPressed = True
            # print("mouse pressed with control: {}".format(self.controlPressed))
        else:
            self.controlPressed = False
            # print("mouse pressed: {}".format(self.controlPressed))
        self.update()
        QWidget.mousePressEvent(self, ev)

    def mouseReleaseEvent(self, ev):
        # print("mouse released: {}".format(self.controlPressed))
        self.update()
        QWidget.mouseReleaseEvent(self, ev)
