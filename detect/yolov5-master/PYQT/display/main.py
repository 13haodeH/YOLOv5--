#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/26 19:36
# @Author  : 沐白
# @Site    :
# @File    : Main.py
# @Software: PyCharm


# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow
# import asa
#
#
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     MainWindow = QMainWindow()
#     ui = asa.Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())


import sys
import DisplayUI
from PyQt5.QtWidgets import QApplication, QMainWindow
from VideoDisplay import Display

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWnd = QMainWindow()
    ui = DisplayUI.Ui_MainWindow()
    # 可以理解成将创建的 ui 绑定到新建的 mainWnd 上
    ui.setupUi(mainWnd)
    display = Display(ui, mainWnd)
    mainWnd.show()

    sys.exit(app.exec_())
