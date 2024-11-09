# -*- coding: utf-8 -*-

import sys, csv
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from typing import Dict, List


class ResultListModel(QStandardItemModel):
    """결과 리스트 모델"""

    HEADERS: Dict[str, str] = {
        "filename": "파일명",
        "ground_truth": "정답",
        "prediction": "예측",
        "score": "점수",
    }
    __dir_tree: Dict[str, QStandardItem] = dict()

    def __init__(self):
        super().__init__()
        self.setHorizontalHeaderLabels(self.HEADERS.values())

    def addItemBeforeDiag(self, itemData: Dict[str, str], insert=False):
        """데이터 라인 추가 (동기)"""
        data_row = []
        self.setRowCount(self.rowCount())
        for key in self.HEADERS.keys():
            if key in itemData:
                data_row.append(QStandardItem(itemData[key]))
            else:
                data_row.append(QStandardItem("-"))
        data_row[0].setToolTip(itemData["path"])
        self.appendRow(data_row)

    def extendItems(self, items: List[Dict[str, str]]):
        """데이터 라인 추가 (비동기)"""
        for itemData in items:
            current_row = self.rowCount()
            self.setRowCount(current_row + 1)
            for key in self.HEADERS.keys():
                if key in itemData:
                    item = QStandardItem(itemData[key])
                else:
                    item = QStandardItem("-")
                if key == "filename":
                    item.setToolTip(itemData["path"])
                self.setItem(
                    current_row,
                    list(self.HEADERS.keys()).index(key),
                    item,
                )

    def deleteResult(self, idx):
        """결과 삭제"""
        self.removeRow(idx)

    def deleteAllItem(self):
        """모든 결과 삭제"""
        for i in range(self.rowCount() - 1, -1, -1):
            self.removeRow(i)
