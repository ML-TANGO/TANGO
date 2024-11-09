from pathlib import Path
from typing import List

from PyQt5 import QtGui, QtCore

from ..Utils import getFilePath, getFileName, makeDirectory, isFileExist


class ImageLoadWorker(QtCore.QThread):
    """이미지 로드 워커 쓰레드"""

    initLoadStatus = QtCore.pyqtSignal(int)
    processLoadStatus = QtCore.pyqtSignal(int)
    addIamgeFile = QtCore.pyqtSignal(dict)
    workerFinished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.__imageData: List[dict] = list()

    @property
    def imageData(self) -> List[dict]:
        """이미지 데이터 리스트 반환"""
        return self.__imageData

    def start(self, image_dir: str):
        """이미지 로드 시작"""
        self._image_dir = image_dir
        super().start()

    def run(self):
        """이미지 로드 실행"""
        image_dir = Path(self._image_dir)
        qt_iamge_formats = QtGui.QImageReader.supportedImageFormats()
        imageExts = [f"*.{bytes(fmt).decode()}" for fmt in qt_iamge_formats]

        imageFilesTemp: List[Path] = list()
        for imageExt in imageExts:
            filesWithExt = image_dir.rglob(imageExt)
            imageFilesTemp.extend(list(filesWithExt))

        imageFiles: List[Path] = list()
        for imageFile in imageFilesTemp:
            if not "thumbs" in imageFile.as_posix():
                imageFiles.append(imageFile)

        self.__imageData.clear()
        imageFiles = sorted(imageFiles)
        self.initLoadStatus.emit(len(imageFiles))
        n = 0
        for imageFile in imageFiles:
            filePath = imageFile.as_posix()
            pathOfFile = getFilePath(filePath)
            nameOfFile = getFileName(filePath)
            thumbPath = "{}/{}".format(pathOfFile, "thumbs")
            makeDirectory(thumbPath)
            thumbFile = "{}/{}".format(thumbPath, nameOfFile)
            if not isFileExist(thumbFile):
                pixmap = QtGui.QPixmap(imageFile.as_posix())
                pixmap = pixmap.scaled(100, 100, QtCore.Qt.KeepAspectRatio)
                pixmap.save(thumbFile)
            pixmap = QtGui.QPixmap(thumbFile)
            data = {
                "filename": imageFile.name,
                "dirpath": image_dir.as_posix(),
                "path": imageFile.as_posix(),
                "ground_truth": imageFile.parent.stem,
                "qpixmap": pixmap,
            }
            self.__imageData.append(data)
            n += 1
            self.processLoadStatus.emit(n)
            QtCore.QCoreApplication.processEvents()
        self.processLoadStatus.emit(0)
        self.workerFinished.emit()
        print("Finished...")
