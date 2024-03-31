# coding=utf-8
from PySide2.QtGui import QIcon, QPixmap
from PySide2.QtWidgets import QMessageBox

class MsgSuccess(QMessageBox):
    def __init__(self):
        super(MsgSuccess, self).__init__()
        self.setWindowTitle('注意')
        self.setWindowIcon(QIcon('img/yologo.png'))
        pix = QPixmap('img/success.svg').scaled(48, 48)
        self.setIconPixmap(pix)
        self.setStyleSheet('font: 16px Microsoft YaHei')


class MsgWarning(QMessageBox):
    def __init__(self):
        super(MsgWarning, self).__init__()
        self.setWindowTitle('注意')
        self.setWindowIcon(QIcon('img/yologo.png'))
        pix = QPixmap('img/warn.svg').scaled(48, 48)
        self.setIconPixmap(pix)
        self.setStyleSheet('font: 16px Microsoft YaHei')
