from PyQt5 import QtWidgets, QtGui
import sys
from imagrec import Ui_ImagRec
from PyQt5.QtCore import QTimer
from datetime import datetime
import random
from multiprocessing import Process
import os
# exit()

class Main():
    def recongnize(self):
        os.system('python usemodel_alexnet.py')

class mywindow(QtWidgets.QWidget, Ui_ImagRec):

    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        '''时间参数设置'''
        self.Runtime.setText('0:00:00')    
        self.timerCal = QTimer()  # 初始化一个定时器
        self.timerCal.setInterval(1000)
        '''计数器标记'''
        self.i = 1
        '''识别主进程'''
        main = Main()
        self.rec = Process(target=main.recongnize, args = ())


    # 计时
    def operate(self):
        self.Runtime.setText(str((datetime.now() - time_init))[0:7])

    def timer(self):
        if self.i == 1:
            global time_init
            time_init = datetime.now()
   
            self.timerCal.start()  # 设置计时间隔并启动
            self.timerCal.timeout.connect(self.operate)  # 计时结束调用operate()方法

    def changeLabel(self):
        if self.i %2 == 1:
            self.pushButton.setText("暂停")
        elif self.i %2 == 0:
            self.pushButton.setText("开始识别")
        self.i += 1

    def recongnize(self):
        if self.i %2 == 0:
            self.rec.start()
        else:
            self.rec.terminate()
            os.system("taskkill/im python.exe -f")
            main = Main()
            self.rec = Process(target=main.recongnize, args = ())

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # 窗口
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
