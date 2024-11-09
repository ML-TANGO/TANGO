import time
from typing import List, Tuple

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from .models import DenseNet201, ResNet152


class DiagThread(QThread):
    diagSignalToGui = pyqtSignal(int, bool, float)
    finishSignalToGui = pyqtSignal()

    def __init__(self, parent):
        QThread.__init__(self, parent)
        self.parent = parent

    def start(self, imageFiles: List[str]):
        self.imageFiles = imageFiles
        super().start()

    def run(self):
        # self.model = DenseNet201() # DenseNet201 모델 사용
        self.model = ResNet152()  # ResNet152 모델 사용
        for i in range(len(self.imageFiles)):
            result = self.diagImage(self.imageFiles[i])
            self.diagSignalToGui.emit(i, *result)
        self.finishSignalToGui.emit()
        del self.model

    def diagImage(self, imageFile: str) -> Tuple[bool, float]:
        try:
            result, score = self.model(imageFile)
        except Exception as e:
            print(e)
            return False
        return result == "Pneumonia", score
