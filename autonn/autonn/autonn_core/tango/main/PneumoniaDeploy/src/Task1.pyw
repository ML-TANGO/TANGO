import os
import sys
from pathlib import Path
from typing import List


from PyQt5 import QtCore, QtGui, QtWidgets

from .theme.qsshelper import QSSHelper
from .DataModelResult import ResultListModel
from .MainWindow import Ui_MainWindow
from .Utils import getJsonDataFromFile, saveJsonData
from .GuiUtils import getFileDirectory, openFileDialog
from .ThumbnailWidget import ThumbnailListWidgetItem, ThumbnailListWidget
from .DiagThread import DiagThread

from .worker import ImageLoadWorker


class AppMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """메인 윈도우 구현 클래스"""

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        self.buttonOpenImage.hide()
        self.progressBar.setValue(0)

        self.readAppSettings()
        self.buttonOpenFolder.clicked.connect(self.onClickOpenFolder)
        self.buttonOpenImage.clicked.connect(self.onClickOpenImage)
        self.buttonDeleteImage.clicked.connect(self.onClickDeleteImage)
        self.buttonDeleteAllImages.clicked.connect(self.onClickDelteAllImages)
        self.listImages.itemClicked.connect(self.onItemClickListImages)
        self.listResult.clicked.connect(self.onClickListResult)
        self.buttonSetToZoomView.clicked.connect(self.onClickSetToZoomView)
        self.buttonSetToFitView.clicked.connect(self.onClickSetToFitView)
        self.buttonSaveResult.clicked.connect(self.onClickSaveResult)
        self.buttonStart.clicked.connect(self.onClickStart)
        self.setupListWidget()
        self.setWinRect()

        self.initResult()

        self.selectedIndex = -1

        self.show()

        self._imageLoadWorker = ImageLoadWorker()
        self._imageLoadWorker.initLoadStatus.connect(self.initLoadStatus)
        self._imageLoadWorker.processLoadStatus.connect(self.processLoadStatus)
        self._imageLoadWorker.addIamgeFile.connect(self.addImageToImageList)
        self._imageLoadWorker.workerFinished.connect(self.finishImageLoad)
        self.listResult.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.diagThread = DiagThread(self)
        self.diagThread.diagSignalToGui.connect(self.onRecvResultFromDiagThread)
        self.diagThread.finishSignalToGui.connect(self.onFinishFromDiagThread)
        self.progressBar.setVisible(False)
        self.buttonStart.setVisible(False)
        self.buttonClearResult.setVisible(False)

    def closeEvent(self, event):
        """프로그램 종료 이벤트"""
        self.saveWinRect()

    def readAppSettings(self):
        """앱 설정 파일을 읽어서 변수에 저장"""
        self.appSettings = getJsonDataFromFile("./appSettings.json")

    def initResult(self):
        """진단 결과 테이블 초기화"""
        self.dataModelResult = ResultListModel()
        self.listResult.setModel(self.dataModelResult)

    def onClickOpenFolder(self):
        """디렉토리 내 이미지 파일을 리스트에 추가"""
        folder = getFileDirectory(self, "폴더를 선택하세요.")
        if folder != None:
            self.progressBar.setVisible(True)
            self.addFolderToImageList(folder)

    def onClickOpenImage(self):
        """이미지 파일을 리스트에 추가"""
        count, images = openFileDialog(
            self,
            "이미지를 선택하세요.",
            ".",
            "All Files(*);; Jpeg(*.jpg *.jpeg) ;; PNG(*.png)",
        )
        if count > 0:
            for image in images:
                self.addImageToImageList(image)

    def onClickDeleteImage(self):
        """선택된 이미지 삭제"""
        items = self.listImages.selectedItems()
        for item in items:
            itemRow = self.listImages.row(item)
            deleted = self.listImages.takeItem(itemRow)
            del deleted
            self.dataModelResult.deleteResult(itemRow)
        self.updateImageCount()

    def onClickDelteAllImages(self):
        """모든 이미지 삭제"""
        self.listImages.clear()
        self.dataModelResult.deleteAllItem()
        self.selectedIndex = -1
        self.labelSelectedImageName.setText("")
        self.labelSelectedImage.clearImage()
        self.updateImageCount()

    def onClickSaveResult(self):
        """결과 저장"""
        print("save result...")
        result_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Result", "", "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if result_path.endswith(".csv"):
            self.saveToCSV(result_path)
        elif result_path.endswith(".xlsx"):
            self.saveToExcel(result_path)

    def saveToCSV(self, file_path):
        """CSV 파일로 저장"""
        import csv

        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            headers = [header for header in self.dataModelResult.HEADERS.values()]
            writer.writerow(headers)

            for row in range(self.dataModelResult.rowCount()):
                data = []
                for col, key in enumerate(self.dataModelResult.HEADERS.keys()):
                    item = self.dataModelResult.item(row, col)
                    if key == "prediction" and item.text() == "-":
                        break
                    data.append(item.text())
                else:
                    writer.writerow(data)

    def saveToExcel(self, file_path):
        """엑셀 파일로 저장"""
        import openpyxl

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append([header for header in self.dataModelResult.HEADERS.values()])

        for row in range(self.dataModelResult.rowCount()):
            data = []
            for col, key in enumerate(self.dataModelResult.HEADERS.keys()):
                item = self.dataModelResult.item(row, col)
                if key == "prediction" and item.text() == "-":
                    break
                data.append(item.text())
            else:
                sheet.append(data)

        workbook.save(file_path)

    def setupListWidget(self):
        """리스트 위젯 설정"""
        self.listImages.setViewMode(QtWidgets.QListWidget.ListMode)
        self.listImages.setIconSize(QtCore.QSize(100, 100))
        self.listImages.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.listImages.setWrapping(True)
        self.listImages.setWordWrap(True)
        self.listImages.setResizeMode(QtWidgets.QListWidget.Adjust)

    def initLoadStatus(self, countOfAllImages):
        """이미지 로드 상태 초기화"""
        self.progressBar.setMaximum(countOfAllImages)
        self.progressBar.setValue(0)

    def processLoadStatus(self, loadIndex):
        """이미지 로드 상태 진행"""
        self.progressBar.setValue(loadIndex)

    def addImageToImageList(self, imageData: dict):
        """이미지를 리스트에 추가"""
        self.dataModelResult.addItemBeforeDiag(imageData)
        fpath = Path(imageData["path"])
        listThumbnail = ThumbnailListWidget()
        listThumbnail.setTextDown(listThumbnail.makeShortName(fpath.stem))
        listThumbnail.setIcon(imageData.get("qpixmap", QtCore.QPixmap()))
        thumbnailItemItem = ThumbnailListWidgetItem(self.listImages)
        thumbnailItemItem.setSizeHint(listThumbnail.sizeHint())
        thumbnailItemItem.setImageInfo(fpath.stem, fpath.as_posix())
        thumbnailItemItem.setToolTip(fpath.as_posix())
        self.listImages.addItem(thumbnailItemItem)
        self.listImages.setItemWidget(thumbnailItemItem, listThumbnail)

        self.updateImageCount()

    def finishImageLoad(self):
        """이미지 로드 완료 후 처리"""
        print("finished image loading...")
        iamgeData = self._imageLoadWorker.imageData
        self.listImages.clear()
        self.dataModelResult.deleteAllItem()
        self.dataModelResult.blockSignals(True)
        self.dataModelResult.extendItems(iamgeData)
        self.dataModelResult.blockSignals(False)
        self.listResult.update()
        self.listResult.setCurrentIndex(self.dataModelResult.index(0, 0))

        self.buttonStart.setVisible(True)
        self.progressBar.setVisible(False)

        for data in iamgeData:
            fpath = Path(data["path"])
            listThumbnail = ThumbnailListWidget()
            listThumbnail.setTextDown(listThumbnail.makeShortName(fpath.stem))
            listThumbnail.setIcon(data.get("qpixmap", QtGui.QPixmap()))
            thumbnailItemItem = ThumbnailListWidgetItem(self.listImages)
            thumbnailItemItem.setSizeHint(listThumbnail.sizeHint())
            thumbnailItemItem.setImageInfo(fpath.stem, fpath.as_posix())
            thumbnailItemItem.setToolTip(fpath.as_posix())
            self.listImages.addItem(thumbnailItemItem)
            self.listImages.setItemWidget(thumbnailItemItem, listThumbnail)
            QtCore.QCoreApplication.processEvents()

        self.updateImageCount()

    def updateImageCount(self):
        """이미지 카운트 업데이트"""
        cnt = self.listImages.count()
        self.labelImageCount.setText("{} Files".format(cnt))

    def addFolderToImageList(self, path):
        """폴더 내 이미지를 리스트에 추가"""
        self._imageLoadWorker.start(path)

    def onItemClickListImages(self):
        """이미지 리스트에서 아이템 클릭시 처리"""
        self.setCurrentImage()
        key = self.listImages.currentItem().toolTip()
        item = None
        for i in range(self.dataModelResult.rowCount()):
            if self.dataModelResult.item(i, 0).toolTip() == key:
                item = self.dataModelResult.item(i, 0)
                break
        if item:
            self.listResult.setCurrentIndex(item.index())
        else:
            self.listResult.clearSelection()
        return

    def setCurrentImage(self):
        """선택된 이미지를 표시"""
        selItem = self.listImages.currentItem()
        try:
            imagePath = selItem.getImageInfo()[1]
            self.labelSelectedImage.setImageFromFile(imagePath)
            self.labelSelectedImageName.setText(imagePath)
        except Exception as e:
            print(e)

    def setCurrentResult(self, rowIndex):
        """선택된 결과를 표시"""
        self.listResult.setCurrentIndex(self.dataModelResult.index(rowIndex, 0))

    def onClickListResult(self):
        """결과 리스트 클릭시 처리"""
        indexes = self.listResult.selectedIndexes()
        if indexes:
            self.selectedIndex = indexes[0].row()
            self.listImages.setCurrentRow(self.selectedIndex)
            self.setCurrentImage()

    def onClickSetToZoomView(self):
        """이미지 확대"""
        self.labelSelectedImage.setToZoomView()

    def onClickSetToFitView(self):
        """이미지 축소"""
        self.labelSelectedImage.setToFitView()

    def onClickStart(self):
        """진단 시작"""
        imageFiles = []
        self.__image_result = list()
        self.__result_item_list: List[QtGui.QStandardItem] = list()
        for i in range(self.dataModelResult.rowCount()):
            item = self.dataModelResult.item(i, 0)
            imageFiles.append(item.toolTip())
            self.__result_item_list.append(item)
        if len(imageFiles) == 0:
            QtWidgets.QMessageBox.warning(
                self, "경고", "진단할 이미지를 선택하지 않았습니다."
            )
            return
        self.progressBar.show()
        self.progressBar.setMaximum(len(imageFiles) - 1)
        self.progressBar.setValue(0)
        self.diagThread.start(imageFiles)
        self.buttonStart.hide()

    def onRecvResultFromDiagThread(
        self, imageIndex: int, prediction: bool, score: float
    ):
        """진단 결과 수신"""
        self.progressBar.setValue(imageIndex)
        result_item_root = self.__result_item_list[imageIndex]
        model_index = self.dataModelResult.indexFromItem(result_item_root)
        prediction = "Pneumonia" if prediction else "Normal"
        self.dataModelResult.setData(model_index.siblingAtColumn(2), prediction)
        self.dataModelResult.setData(model_index.siblingAtColumn(3), score)

        gt_data = self.dataModelResult.data(model_index.siblingAtColumn(1))
        self.__image_result.append((gt_data.lower(), prediction.lower()))
        imageCount = len(self.__image_result)
        success = sum([1 for gt, pred in self.__image_result if gt == pred])
        failure = imageCount - success
        accuracy = success / imageCount * 100
        self.labelSuccess.setText(f"{success:,}")
        self.labelFailure.setText(f"{failure:,}")
        self.labelWorked.setText(f"{imageCount:,}")
        self.labelAccuracy.setText(f"{accuracy:.2f} %")

    def onFinishFromDiagThread(self):
        """진단 쓰레드 종료"""
        print("Diag Thread was finished...")
        self.progressBar.hide()
        self.buttonStart.show()

    def setWinRect(self):
        """윈도우 위치 설정"""
        self.winLeft = self.appSettings["Geometry"]["Left"]
        self.winTop = self.appSettings["Geometry"]["Top"]
        self.winWidth = self.appSettings["Geometry"]["Width"]
        self.winHeight = self.appSettings["Geometry"]["Height"]
        self.resize(self.winWidth, self.winHeight)
        self.move(self.winLeft, self.winTop)
        self.splitter_2.setSizes(self.appSettings["Geometry"]["IMAGE_RESULT"])
        self.splitter.setSizes(self.appSettings["Geometry"]["IMAGE_LIST"])

    def saveWinRect(self):
        """윈도우 위치 저장"""
        self.getWinRect()
        saveJsonData("./appSettings.json", self.appSettings)

    def getWinRect(self):
        """윈도우 위치 정보 저장"""
        rc = self.frameGeometry()
        self.appSettings["Geometry"]["Left"] = rc.left()
        self.appSettings["Geometry"]["Top"] = rc.top()
        self.appSettings["Geometry"]["Width"] = rc.width()
        self.appSettings["Geometry"]["Height"] = rc.height()
        sizeH = self.splitter_2.sizes()
        self.appSettings["Geometry"]["IMAGE_RESULT"] = sizeH
        sizeV2 = self.splitter.sizes()
        self.appSettings["Geometry"]["IMAGE_LIST"] = sizeV2


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    appPath = os.path.dirname(os.path.abspath("__file__"))

    mainWindow = AppMainWindow()
    app.installEventFilter(mainWindow)
    qss = QSSHelper.open_qss(os.path.join("theme", "githubblue.qss"))

    mainWindow.setStyleSheet(qss)
    sys.exit(app.exec_())
