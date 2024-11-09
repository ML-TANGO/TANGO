# -*- coding: utf-8 -*-
"""어플리케이션 실행
"""
import os
import sys

from PyQt5.QtWidgets import QApplication

from src import AppMainWindow
from src.theme.qsshelper import QSSHelper


if __name__ == "__main__":
    app = QApplication(sys.argv)
    appPath = os.path.dirname(os.path.abspath("__file__"))
    mainWindow = AppMainWindow()
    app.installEventFilter(mainWindow)
    qss = QSSHelper.open_qss(os.path.join("src", "theme", "githubblue.qss"))
    mainWindow.setStyleSheet(qss)
    sys.exit(app.exec_())
