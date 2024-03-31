import sys
import untitled
from PyQt5.QtWidgets import QApplication,QMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = untitled.Ui_MainWindow()
    ##向主窗口中添加控件
    ui.setupUi(MainWindow)
    MainWindow.show()
    ##进入主循环
    sys.exit(app.exec_())