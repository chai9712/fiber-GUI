# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:11:07 2019

@author: Administrator
"""

import tensorflow as tf
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import MLP_test as mt
import LSTM_test as lt
import NSGA2_train as nt

currentTime = ""

input_avg=[]
input_avg.append([8.442307692,7.999216518,35.04326923,403.8967827])
input_avg.append([41.20192308,10.37179144	,509.1708212])
input_avg.append([47.69230769,10.36493655,530.3418788])
input_avg.append([53.43269231,10.34451232,545.2828404])
input_avg.append([61.94230769,10.28733596,602.9103404])
input_avg.append([65.67307692,	10.25060709,606.5699558])
input_avg.append([65.67307692,7.42828172,737.7392073])
input_avg.append([65.67307692,5.333012184,937.5462748])
input_avg.append([66.15384615,3.630155716,1264.498739])
input_avg.append([25.32115385,1.77959255,	1212.600991])
input_avg.append([149.6634615,4.907126572,1260.736183])
input_avg.append([0.442403846,1.160355227,3672.505192])
input_avg.append([0.118942308,0.727403383,4130.762308])
input_avg.append([25,0.486326593,3960.498269])
input_avg.append([140,1.678577731,	3980.900673])

input_res=[]
input_res.append([0,0,0,0])
input_res.append([0,0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])
input_res.append([0,0,0])

inputV=[]#初始8个参数和后14个炉子的参数
in_num_c=8
in_num_x=42
in_num=50
out_num=10
batch_size = 3
res_write=[]

#读取LSTM模型
graph_lstm=tf.Graph()
with graph_lstm.as_default():  
    saver_lstm = tf.train.import_meta_graph("./Model_LSTM/LSTM_8_42_10.meta")# 注意此处路径前添加"./"  
    sess_lstm=tf.Session(graph=graph_lstm)#创建新的sess
with sess_lstm.as_default():
     with graph_lstm.as_default():
        saver_lstm.restore(sess_lstm,tf.train.latest_checkpoint("./Model_LSTM/"))#从恢复点恢复参数
#读取MLP模型
graph_mlp=tf.Graph()
with graph_mlp.as_default():  
    saver_mlp = tf.train.import_meta_graph("./Model_MLP/UniMLP_50_10.meta")# 注意此处路径前添加"./"  
    sess_mlp=tf.Session(graph=graph_mlp)#创建新的sess
with sess_mlp.as_default():
     with graph_mlp.as_default():
        saver_mlp.restore(sess_mlp,tf.train.latest_checkpoint("./Model_MLP/"))#从恢复点恢复参数


class aboutUI(QMainWindow):
    def __init__(self, parent=None):
        super(aboutUI, self).__init__(parent)
        
        self.setFixedSize(400,250)
        font = QFont()
        font.setFamily("Aharoni")
        font.setPointSize(10)
        font.setBold(True)
        self.centralwidget = QWidget()
        
        self.lbl = QLabel(self.centralwidget)
        self.lbl.setFont(font)
        self.btn1 = QPushButton(self.centralwidget)
        self.btn2 = QPushButton(self.centralwidget)
        self.btn3 = QPushButton(self.centralwidget)
        self.btn4 = QPushButton(self.centralwidget)

        self.lbl.setText("本程序可实现对碳纤维生产工艺参数进行预测，共有三个功能")
        self.btn1.setText("正向（MLP）  ")
        self.btn2.setText("正向（LSTM） ")
        self.btn3.setText("逆向（NSGA2）")
        self.btn4.setText("    返回     ")
        
        self.btn1.clicked.connect(lambda:self.openmsg(self.btn1))
        self.btn2.clicked.connect(lambda:self.openmsg(self.btn2))
        self.btn3.clicked.connect(lambda:self.openmsg(self.btn3))
        self.btn4.clicked.connect(lambda:self.openmsg(self.btn4))

        
        self.lbl.move(10,20)
        self.btn1.move(160,50)
        self.btn2.move(160,90)
        self.btn3.move(160,130)
        self.btn4.move(160,170)

        
        self.setWindowTitle("碳纤维仿真系统")
        self.setWindowIcon(QIcon('./images/winicon.jpg'))
        self.setCentralWidget(self.centralwidget)
    def openmsg(self,btn):
        if btn.text() == "正向（MLP）":
           QMessageBox.about(self,"正向（MLP）","1.功能：使用已经训练好的模型进行十五个炉子的输出参数预测。\n"+
             "2.使用方法：首先选择已经训练好的MLP模型用来预测控制参数，对于每个\n"+
             "炉子，输入控制参数后，点击下一个按钮，继续进行下一个炉子的输入。\n"+
             "最终逐步实现十五个炉的输出参数预测。\n"+
             "3.输出：全部参数输入后，预测结果将在界面中打印出来。")
        elif btn.text() == "正向（LSTM）":
            QMessageBox.about(self,"正向（LSTM）","1.功能：使用已经训练好的模型进行十五个炉子的输出参数预测。\n"+
             "2.使用方法：首先选择已经训练好的LSTM模型用来预测控制参数，对于每个\n"+
             "炉子，输入控制参数后，点击下一个按钮，继续进行下一个炉子的输入。\n"+
             "最终逐步实现十五个炉的输出参数预测。\n"+
             "3.输出：全部参数输入后，预测结果将在界面中打印出来。\n"+
             "注：该模型为MLP网络和LSTM网络的混合。")
        elif btn.text() == "逆向（NSGA2）":
            QMessageBox.about(self,"逆向模拟","功能：使用已经训练好的模型进行炉子的控制参数和原丝性能预测。\n"+
             "使用方法：首先选择已经训练好的模型用来预测控制参数，只需要在最\n"+
             "初输入一次输出参数，即可开始训练。通过不断点击下一个来逐步实现\n"+
             "十五个炉子的控制参数和原丝性能预测输出。\n"+
             "输出：所有预测结果都会打印在界面上。")
        elif btn.text() == "返回":
            self.startui = StartUI()
            self.startui.setObjectName("MainWindow")
            self.startui.setStyleSheet("#MainWindow{border-image:url(images/bg.png);}")
            self.startui.show()
            self.close()
           
class processStart(QMainWindow):
    def __init__(self,kind):
        super(processStart, self).__init__()
        self.kind=kind
        self.initUI()

    def initUI(self):

        self.resize(670, 565)
        self.setFixedSize(670,565)
        font = QFont()
        font.setFamily("Aharoni")
        font.setPointSize(10)
        font.setBold(True)
        self.pDoubleValidator = QDoubleValidator()
        self.pDoubleValidator.setDecimals(5)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        self.centralwidget = QWidget()
        self.currentCount = 0
        self.indata = []
        self.outdata = []
        self.leftzone = QWidget(self.centralwidget)
        self.leftlayout = QVBoxLayout(self.leftzone)
        
        self.left_down_zone = QWidget(self.leftzone)
        self.lhlo = QHBoxLayout(self.left_down_zone)
        
        
        self.rightzone = QWidget(self.centralwidget)
        self.rightlayout = QVBoxLayout(self.rightzone)
        
        self.right_top_zone = QWidget(self.rightzone)
        self.rthlo = QHBoxLayout(self.right_top_zone)
        
        self.right_top_zone1 = QWidget(self.right_top_zone)
        self.rtglo1 = QGridLayout(self.right_top_zone1)
        
        self.right_top_zone2 = QWidget(self.right_top_zone)
        self.rtglo2 = QGridLayout(self.right_top_zone2)
        
        self.right_top_zone3 = QWidget(self.right_top_zone)
        self.rtglo3 = QGridLayout(self.right_top_zone3)
        
        self.right_down_zone = QWidget(self.rightzone)
        self.rdvlo = QVBoxLayout(self.right_down_zone)
        
        #左边区域
        # 创建滚动条
        self.s1 = QScrollBar()
        self.s1.setMaximum(12)
        self.s1.valueChanged.connect(self.sliderval)
        
     
        self.item = []
        self.msglbl1 = QLabel("碳化炉流程")
        self.msglbl1.setFont(font)
        self.msglbl1.setAlignment(Qt.AlignCenter)
        self.imglbl = QLabel()
        
        for i in range(14):
            
            self.item.append(QPushButton("碳化炉"+str(i+1))) 
            imglbl = QLabel()
            imglbl.setPixmap(QPixmap("./images/arrowblue1.png"))
            imglbl.setAlignment(Qt.AlignCenter)
            self.item.append(imglbl)
            #self.item.append(QLabel("L"))
        self.item.append(QPushButton("碳化炉15"))
        self.item[1].setPixmap(QPixmap("./images/arrowred1.png"))
        self.btngroup = QWidget()
        self.lglo = QGridLayout(self.btngroup)
        for i in range(16):
            self.lglo.addWidget(self.item[i],i,0,1,1)
        for i in range(28, 15,-1):
            self.lglo.addWidget(self.item[i],16,0,1,1)
        
        #右边区域right_top_zone1
        self.temperatureLbl = QLabel("温度：")
        self.timeLbl = QLabel("时间：")
        self.speedLbl = QLabel("速度：")
        self.concentrationLbl = QLabel("浓度：")
        
        self.msglbl2 = QLabel("参数输入")
        self.msglbl2.setFont(font)
        
        
        self.temperatureEdit = QLineEdit()
        self.timeEdit = QLineEdit()
        self.speedEdit = QLineEdit()
        self.concentrationEdit = QLineEdit()
        
        self.temperatureEdit.setValidator(self.pDoubleValidator)
        self.timeEdit.setValidator(self.pDoubleValidator)
        self.speedEdit.setValidator(self.pDoubleValidator)
        self.concentrationEdit.setValidator(self.pDoubleValidator)
        
        finputdata = input_avg[0]
        self.temperatureEdit.setText(str(finputdata[0]))
        self.timeEdit.setText(str(finputdata[1]))
        self.speedEdit.setText(str(finputdata[2]))
        self.concentrationEdit.setText(str(finputdata[3]))
        
         #右边区域right_top_zone1
        self.emptyLbl1 = QLabel("")
        self.emptyLbl2 = QLabel("")
        self.rtglo1.addWidget(self.emptyLbl1,0,0,1,1)
        self.rtglo1.addWidget(self.emptyLbl2,0,1,1,1)
        
         #右边区域right_top_zone2
        self.returnbtn = QPushButton("返回")
        self.returnbtn.clicked.connect(self.processReturn)
        
        self.sendbtn = QPushButton("提交参数")
        self.sendbtn.clicked.connect(self.processSend)
        
        self.rtglo2.addWidget(self.msglbl2,0,1,1,1)
        self.rtglo2.addWidget(self.temperatureLbl,1,0,1,1)
        self.rtglo2.addWidget(self.timeLbl,2,0,1,1)
        self.rtglo2.addWidget(self.speedLbl,3,0,1,1)
        self.rtglo2.addWidget(self.concentrationLbl,4,0,1,1)
        self.rtglo2.addWidget(self.returnbtn,5,0,1,1)
        
        self.rtglo2.addWidget(self.temperatureEdit,1,1,1,1)
        self.rtglo2.addWidget(self.timeEdit,2,1,1,1)
        self.rtglo2.addWidget(self.speedEdit,3,1,1,1)
        self.rtglo2.addWidget(self.concentrationEdit,4,1,1,1)
        self.rtglo2.addWidget(self.sendbtn,5,1,1,1)

        #右边区域right_top_zone3
        self.emptyLbl3 = QLabel("           ")
        self.emptyLbl4 = QLabel("           ")
        self.rtglo3.addWidget(self.emptyLbl3,0,0,1,1)
        self.rtglo3.addWidget(self.emptyLbl4,0,1,1,1)
 
        #右边区域下方right_down_zone
        self.msglbl4 = QLabel("模拟过程记录")
        self.msglbl4.setFont(font)
        self.resultTextEdit = QTextEdit()
        self.resultTextEdit.setFixedSize(460,310)
       
        self.rdvlo.addWidget(self.msglbl4)
        self.rdvlo.addWidget(self.resultTextEdit)
     
        # 为布局添加控件
        self.lhlo.addWidget(self.btngroup)
        self.lhlo.addWidget(self.s1)
        
        self.leftlayout.addWidget(self.msglbl1)
        self.leftlayout.addWidget(self.left_down_zone)
        
        self.rthlo.addWidget(self.right_top_zone1)
        self.rthlo.addWidget(self.imglbl)
        self.rthlo.addWidget(self.right_top_zone2)
        
        
        self.rightlayout.addWidget(self.right_top_zone)
        self.rightlayout.addWidget(self.right_down_zone)
        
        self.rightzone.move(150,0)
        
        self.setWindowTitle("碳纤维仿真系统")
        self.setWindowIcon(QIcon('./images/winicon.jpg'))
        self.setCentralWidget(self.centralwidget)
        
       
    def sliderval(self):
        temp = self.s1.value()
        
        for i in range(temp, temp+17):
            self.lglo.addWidget(self.item[i],i-temp,0,1,1)
    
    def processReturn(self):
        self.startui = StartUI()
        self.startui.setObjectName("MainWindow")
        self.startui.setStyleSheet("#MainWindow{border-image:url(images/bg.png);}")
        self.startui.show()
        self.close()
    def processClear(self):
        self.temperatureEdit.setText("")
        self.timeEdit.setText("")
        self.speedEdit.setText("")
        self.concentrationEdit.setText("")
    def processSend(self):
        
        global currentTime
        time = QDateTime.currentDateTime()
        currentTime = time.toString("yyyy-MM-dd hh:mm:ss")
       
        if self.currentCount < 27:
            self.indata = []
            self.indata.append(float(self.temperatureEdit.text()))
            self.indata.append(float(self.timeEdit.text()))
            self.indata.append(float(self.speedEdit.text()))
            if(int(self.currentCount/2)==0):#只有第一个炉子有四个参数
                self.indata.append(float(self.concentrationEdit.text()))
                    
            self.item[self.currentCount].setEnabled(False)
            self.item[self.currentCount+1].setPixmap(QPixmap("./images/arrowgray1.png"))
            if self.currentCount < 26:
                self.item[self.currentCount+3].setPixmap(QPixmap("./images/arrowred1.png"))
            self.s1.setValue(int(self.currentCount/2+1))
            
            if(self.kind==0):
                mt.MlpAppend(int(self.currentCount/2),self.indata)#传递参数
            else:
                lt.LstmAppend(int(self.currentCount/2),self.indata)#传递参数
            if(int(self.currentCount/2)==0):
                self.resultTextEdit.append(currentTime+"\n"+"碳化炉"+ str(int(self.currentCount/2+1))+"号"+"\n"+"输入参数：温度："+str(self.indata[0])+"\t时间："+str(self.indata[1])+
                                       "\n          速度："+str(self.indata[2])+"\t浓度："+str(self.indata[3]))
            else:
                self.resultTextEdit.append(currentTime+"\n"+"碳化炉"+ str(int(self.currentCount/2+1))+"号"+"\n"+"输入参数：温度："+str(self.indata[0])+"\t时间："+str(self.indata[1])+
                                       "\n          速度："+str(self.indata[2]))    
            self.statusBar.showMessage("执行")
            inputdata_temp = input_avg[int(self.currentCount/2)+1]
            self.temperatureEdit.setText(str(inputdata_temp[0]))
            self.timeEdit.setText(str(inputdata_temp[1]))
            self.speedEdit.setText(str(inputdata_temp[2]))
            self.concentrationEdit.setText("")
        
            self.currentCount = self.currentCount + 2;
                
        else:
            self.indata = []
            self.indata.append(float(self.temperatureEdit.text()))
            self.indata.append(float(self.timeEdit.text()))
            self.indata.append(float(self.speedEdit.text()))
            #self.indata.append(float(self.concentrationEdit.text()))
            
            if(self.kind==0):
                mt.MlpAppend(int(self.currentCount/2),self.indata)#传递参数
            else:
                lt.LstmAppend(int(self.currentCount/2),self.indata)#传递参数
            
            self.resultTextEdit.append(currentTime+"\n"+"碳化炉"+ str(int(self.currentCount/2+1))+"号"+"\n"+"输入参数：温度："+str(self.indata[0])+"\t时间："+str(self.indata[1])+
                                       "\n          速度："+str(self.indata[2]))
        
            self.statusBar.showMessage("就绪")
            #获得模拟结果
            global graph_lstm,saver_lstm,sess_lstm
            global graph_mlp,saver_mlp,sess_mlp
            if(self.kind==0):
                y=mt.MlpTest(graph_mlp,saver_mlp,sess_mlp)
            else:
                y=lt.LstmTest(graph_lstm,saver_lstm,sess_lstm)
            
            self.resultTextEdit.append(currentTime+"\n"+"碳纤维性能：")
            self.resultTextEdit.append("L501："+str(y[0])+"\tL502："+ str(y[1]))
            self.resultTextEdit.append("P501："+str(y[2])+"\tS501："+ str(y[3]))
            self.resultTextEdit.append("S502："+str(y[4])+"\tB501："+ str(y[5]))
            self.resultTextEdit.append("M501："+str(y[6])+"\tM502："+ str(y[7]))
            self.resultTextEdit.append("O501："+str(y[8])+"\tR501："+ str(y[9]))
              
            self.item[self.currentCount].setEnabled(False)
            self.sendbtn.setEnabled(False)
            
            QMessageBox.about(self,"成功","模拟成功! ")           
                   
class processEnd(QMainWindow):
    def __init__(self):
        super(processEnd, self).__init__()
        self.initUI()

    def initUI(self):

        self.resize(650, 565)
        self.setFixedSize(650,565)
        font = QFont()
        font.setFamily("Aharoni")
        font.setPointSize(10)
        font.setBold(True)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        #总体布局
        # 中心组件
        self.centralwidget = QWidget()
        self.currentCount = 0
        self.indata = []
        self.outdata = []
        self.leftzone = QWidget(self.centralwidget)
        self.leftlayout = QVBoxLayout(self.leftzone)
        
        self.left_down_zone = QWidget(self.leftzone)
        self.lhlo = QHBoxLayout(self.left_down_zone)
        
        
        self.rightzone = QWidget(self.centralwidget)
        self.rightlayout = QVBoxLayout(self.rightzone)
        
        self.right_top_zone = QWidget(self.rightzone)
        self.rthlo = QHBoxLayout(self.right_top_zone)
        
        self.right_top_zone1 = QWidget(self.right_top_zone)
        self.rtglo1 = QGridLayout(self.right_top_zone1)
        
        self.right_top_zone2 = QWidget(self.right_top_zone)
        self.rtglo2 = QGridLayout(self.right_top_zone2)
        
        self.right_top_zone3 = QWidget(self.right_top_zone)
        self.rtglo3 = QGridLayout(self.right_top_zone3)
        
        self.right_down_zone = QWidget(self.rightzone)
        self.rdvlo = QVBoxLayout(self.right_down_zone)
        
        #左边区域
        # 创建滚动条
        self.s1 = QScrollBar()
        self.s1.setMaximum(12)
        self.s1.valueChanged.connect(self.sliderval)
        
     
        self.item = []
        self.msglbl1 = QLabel("碳化炉流程")
        self.msglbl1.setFont(font)
        self.msglbl1.setAlignment(Qt.AlignCenter)
        self.imglbl = QLabel()
        self.imglbl.setPixmap(QPixmap("./images/right_blue.png"))
    
        for i in range(14):
            self.item.append(QPushButton("碳化炉"+str(15-i))) 
            imglbl = QLabel()
            imglbl.setPixmap(QPixmap("./images/arrowblue1.png"))
            imglbl.setAlignment(Qt.AlignCenter)
            self.item.append(imglbl)
            
            
        self.item.append(QPushButton("碳化炉1"))
        self.item[1].setPixmap(QPixmap("./images/arrowred1.png"))
        self.btngroup = QWidget()
        self.lglo = QGridLayout(self.btngroup)
        for i in range(16):
            self.lglo.addWidget(self.item[i],i,0,1,1)
        for i in range(28, 15,-1):
            self.lglo.addWidget(self.item[i],16,0,1,1)
            
        #右边区域right_top_zone1        
        self.emptyLbl1 = QLabel("      ")
        self.emptyLbl2 = QLabel("      ")
        self.rtglo1.addWidget(self.emptyLbl1,0,0,1,1)
        self.rtglo1.addWidget(self.emptyLbl2,0,1,1,1)
         #右边区域right_top_zone3
        
        self.emptyLbl3 = QLabel("         ")
        self.emptyLbl4 = QLabel("         ")
        self.rtglo3.addWidget(self.emptyLbl3,0,0,1,1)
        self.rtglo3.addWidget(self.emptyLbl4,0,1,1,1)
        #右边区域right_top_zone2
        self.sLbl = QLabel("S：")
        self.oLbl = QLabel("O：")
        self.lLbl = QLabel("L：")
        self.pLbl = QLabel("P：")
        self.returnbtn = QPushButton("返回")
        self.returnbtn.clicked.connect(self.processReturn)
        self.sendbtn = QPushButton("下一步")
        self.sendbtn.clicked.connect(self.processSend)
        self.msglbl3 = QLabel("参数优化")
        self.msglbl3.setFont(font)
        
        
        self.sEdit = QLineEdit()
        #self.sEdit.setText(str(iinresultdata[0]))
        self.sEdit.setReadOnly(True)
        self.oEdit = QLineEdit()
        #self.oEdit.setText(str(iinresultdata[1]))
        self.oEdit.setReadOnly(True)
        self.lEdit = QLineEdit()
        #self.lEdit.setText(str(iinresultdata[2]))
        self.lEdit.setReadOnly(True)
        self.pEdit = QLineEdit()
        self.pEdit.setReadOnly(True)
        
        self.rtglo2.addWidget(self.msglbl3,0,1,1,1)
        self.rtglo2.addWidget(self.sLbl,1,0,1,1)
        self.rtglo2.addWidget(self.oLbl,2,0,1,1)
        self.rtglo2.addWidget(self.lLbl,3,0,1,1)
        self.rtglo2.addWidget(self.pLbl,4,0,1,1)
        
        self.rtglo2.addWidget(self.sEdit,1,1,1,1)
        self.rtglo2.addWidget(self.oEdit,2,1,1,1)
        self.rtglo2.addWidget(self.lEdit,3,1,1,1)
        self.rtglo2.addWidget(self.pEdit,4,1,1,1)
        self.rtglo2.addWidget(self.sendbtn,5,1,1,1)
        self.rtglo2.addWidget(self.returnbtn,5,0,1,1)
 
        #右边区域下方right_down_zone
        self.msglbl4 = QLabel("模拟过程记录")
        self.msglbl4.setFont(font)
        self.resultTextEdit = QTextEdit()
        self.resultTextEdit.setFixedSize(460,310)
        
        self.rdvlo.addWidget(self.msglbl4)
        self.rdvlo.addWidget(self.resultTextEdit)
        #self.rdvlo.addWidget(self.testlbl)
       
        
        # 为布局添加控件
        self.lhlo.addWidget(self.btngroup)
        self.lhlo.addWidget(self.s1)
        
        self.leftlayout.addWidget(self.msglbl1)
        self.leftlayout.addWidget(self.left_down_zone)
        
        self.rthlo.addWidget(self.right_top_zone1)
        self.rthlo.addWidget(self.right_top_zone2)
        self.rthlo.addWidget(self.right_top_zone3)
        
        self.rightlayout.addWidget(self.right_top_zone)
        self.rightlayout.addWidget(self.right_down_zone)
        
        self.rightzone.move(150,0)
    
        self.setWindowTitle("碳纤维仿真系统")
        self.setWindowIcon(QIcon('./images/winicon.jpg'))
        self.setCentralWidget(self.centralwidget)
        
       
    def sliderval(self):
        temp = self.s1.value()
        
        for i in range(temp, temp+17):
            self.lglo.addWidget(self.item[i],i-temp,0,1,1)
    
    def processReturn(self):
        self.startui = StartUI()
        self.startui.setObjectName("MainWindow")
        self.startui.setStyleSheet("#MainWindow{border-image:url(images/bg.png);}")
        self.startui.show()
        self.close()
    def processClear(self):
        self.temperatureEdit.setText("")
        self.timeEdit.setText("")
        self.speedEdit.setText("")
        self.concentrationEdit.setText("")
    def processSend(self):
        global currentTime
        time = QDateTime.currentDateTime()
        currentTime = time.toString("yyyy-MM-dd hh:mm:ss")
        self.statusBar.showMessage("执行")
        
        global input_res,res_write
        global graph_mlp,saver_mlp,sess_mlp
        if(self.currentCount/2==0):
            y=nt.MainIn(graph_mlp,saver_mlp,sess_mlp)        
            count=0
            for i in range(16):
                for j in range(len(input_res[i])):
                    input_res[i][j]=y[count]
                    count=count+1
                
        iinresultdata=input_res[15-int(self.currentCount/2)]
#        iinresultdata=input_avg[14-int(self.currentCount/2)]
        self.sEdit.setText("")
        self.oEdit.setText("")
        self.lEdit.setText("")
        self.pEdit.setText("")
        self.ln = len(iinresultdata)
        if self.ln == 3:
            self.sEdit.setText(str(iinresultdata[0]))
            self.oEdit.setText(str(iinresultdata[1]))
            self.lEdit.setText(str(iinresultdata[2]))
        elif self.ln == 4:
            self.sEdit.setText(str(iinresultdata[0]))
            self.oEdit.setText(str(iinresultdata[1]))
            self.lEdit.setText(str(iinresultdata[2]))
            self.pEdit.setText(str(iinresultdata[3]))
        if self.currentCount < 27:
            self.item[self.currentCount].setEnabled(False)
            self.item[self.currentCount+1].setPixmap(QPixmap("./images/arrowgray1.png"))
            if self.currentCount < 26:
                self.item[self.currentCount+3].setPixmap(QPixmap("./images/arrowred1.png"))
            self.s1.setValue(int(self.currentCount/2+1))
            
            if self.ln == 3:    
                self.resultTextEdit.append(currentTime+"\n"+"碳化炉"+ str(int(15-self.currentCount/2))+"号"+"\n"+"结果参数：S："+str(self.sEdit.text())+"\tO："+str(self.oEdit.text())+
                                           "\n          L："+str(self.lEdit.text())+"\n")
            elif self.ln == 4:
                  self.resultTextEdit.append(currentTime+"\n"+"碳化炉"+ str(int(15-self.currentCount/2))+"号"+"\n"+"结果参数：S："+str(self.sEdit.text())+"\tO："+str(self.oEdit.text())+
                                           "\n          L："+str(self.lEdit.text())+"\tP："+str(self.pEdit.text())+"\n")
                        
            
            self.currentCount = self.currentCount + 2;
            
        else:
            iinresultdata=input_res[1]
#            iinresultdata=input_avg[0]
            self.sEdit.setText(str(iinresultdata[0]))
            self.oEdit.setText(str(iinresultdata[1]))
            self.lEdit.setText(str(iinresultdata[2]))
            self.pEdit.setText(str(iinresultdata[3]))
            self.resultTextEdit.append(currentTime+"\n"+"碳化炉"+ str(int(15-self.currentCount/2))+"号"+"\n"+"结果参数：S："+str(self.sEdit.text())+"\tO："+str(self.oEdit.text())+
                                           "\n          L："+str(self.lEdit.text())+"\tP："+str(self.pEdit.text())+"\n")
            iinresultdata=input_res[0]
#            iinresultdata=np.array([10.6021,334.762577,21.91120288,165.5933966])
            self.resultTextEdit.append("原丝性能：")
            self.resultTextEdit.append("粘均分子量："+str(iinresultdata[0])+"\t落球黏度："+str(iinresultdata[1]))
            self.resultTextEdit.append("特性黏度："+str(iinresultdata[2])+"\t动力黏度："+str(iinresultdata[3]))
            
            self.item[self.currentCount].setEnabled(False)
            self.sendbtn.setEnabled(False)
            QMessageBox.about(self,"成功","模拟成功! ")
#            

#正向MLP和LSTM初始参数
class InputStartUI(QMainWindow):
    def __init__(self,kind, parent=None):
        super(InputStartUI, self).__init__(parent)
        self.kind=kind
        self.oringinData = [0 for i in range(4)]
        self.resize(623, 368)
        self.setFixedSize(423,368)
        self.centralwidget = QWidget()
        self.vlo = QVBoxLayout( self.centralwidget)
        
        if(self.kind==0):
            self.informationlbl = QLabel("原丝参数(MLP)")
        else:
            self.informationlbl = QLabel("原丝参数(LSTM)")
            
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.informationlbl.sizePolicy().hasHeightForWidth())
        self.informationlbl.setSizePolicy(sizePolicy)
        font = QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.informationlbl.setFont(font)
        self.informationlbl.setAlignment(Qt.AlignCenter)
        
        self.vlo.addWidget(self.informationlbl)
    
        InitD= np.array([11.60780221,340.7625577,22.91025288,162.5393942])#均值
        
        self.datas = QWidget()
        self.pDoubleValidator = QDoubleValidator(self.datas)
        self.pDoubleValidator.setDecimals(5)
        self.glo = QGridLayout(self.datas)
        #第一行
        self.lbl1 = QLabel("粘均分子量: ",self.datas)
        self.glo.addWidget(self.lbl1,0,0,1,1)
        self.edit1 = QLineEdit(self.datas)
        self.edit1.setText(str(InitD[0]))
        self.glo.addWidget(self.edit1,0,1,1,1)
        
        self.lbl2 = QLabel("落球黏度：",self.datas)
        self.glo.addWidget(self.lbl2,0,2,1,1)
        self.edit2 = QLineEdit(self.datas)
        self.edit2.setText(str(InitD[1]))
        self.glo.addWidget(self.edit2,0,3,1,1)
        
        #第二行
        self.lbl3 = QLabel("特性黏度：",self.datas)
        self.glo.addWidget(self.lbl3,1,0,1,1)
        self.edit3 = QLineEdit(self.datas)
        self.edit3.setText(str(InitD[2]))
        self.glo.addWidget(self.edit3,1,1,1,1)
        
        self.lbl4 = QLabel("动力黏度：",self.datas)
        self.glo.addWidget(self.lbl4,1,2,1,1)
        self.edit4 = QLineEdit(self.datas)
        self.edit4.setText(str(InitD[3]))
        self.glo.addWidget(self.edit4,1,3,1,1)
        
        self.edit1.setValidator(self.pDoubleValidator)
        self.edit2.setValidator(self.pDoubleValidator)
        self.edit3.setValidator(self.pDoubleValidator)
        self.edit4.setValidator(self.pDoubleValidator)
        
        self.returnbtn = QPushButton("返回")
        self.returnbtn.clicked.connect(self.processReturn)
        self.glo.addWidget(self.returnbtn,5,2,1,2)
        
        self.startbtn = QPushButton("模拟")
        self.startbtn.clicked.connect(self.startTest)
        self.glo.addWidget(self.startbtn,5,0,1,2)
        
        self.vlo.addWidget(self.datas)
        
        
        self.setCentralWidget(self.centralwidget)
        self.setWindowTitle("碳纤维仿真系统")
        self.setWindowIcon(QIcon('./images/winicon.jpg'))    
           
    def startTest(self):
        
        self.oringinData[0] = float(self.edit1.text())
        self.oringinData[1] = float(self.edit2.text())
        self.oringinData[2] = float(self.edit3.text())
        self.oringinData[3] = float(self.edit4.text())
        
        if(self.kind==0):
            mt.MlpAppend(0,self.oringinData)#传递参数
            self.testStart = processStart(0)
        else:
            lt.LstmAppend(0,self.oringinData)#传递参数
            self.testStart = processStart(1)
        self.testStart.show()
        self.close()
        
    def processReturn(self):
        self.startui = StartUI()
        self.startui.setObjectName("MainWindow")
        self.startui.setStyleSheet("#MainWindow{border-image:url(images/bg.png);}")
        self.startui.show()
        self.close()
        
        
#逆向初始参数        
class InputEndUI(QMainWindow):
    def __init__(self, parent=None):
        super(InputEndUI, self).__init__(parent)
        
        self.resultData = [0 for i in range(10)]
        self.resize(500, 368)
        self.setFixedSize(500,368)
        self.centralwidget = QWidget()
        self.vlo = QVBoxLayout( self.centralwidget)
        
        self.informationlbl = QLabel("碳纤维参数")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.informationlbl.sizePolicy().hasHeightForWidth())
        self.informationlbl.setSizePolicy(sizePolicy)
        font = QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.informationlbl.setFont(font)
        self.informationlbl.setAlignment(Qt.AlignCenter)
        
        self.vlo.addWidget(self.informationlbl)

        self.datas = QWidget()
        self.pDoubleValidator = QDoubleValidator(self.datas)
        self.pDoubleValidator.setDecimals(5)
        self.glo = QGridLayout(self.datas)
        #第一行
        self.lbl1 = QLabel("L501: ",self.datas)
        self.glo.addWidget(self.lbl1,0,0,1,1)
        self.edit1 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit1,0,1,1,1)
        
        self.lbl2 = QLabel("L502：",self.datas)
        self.glo.addWidget(self.lbl2,0,2,1,1)
        self.edit2 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit2,0,3,1,1)
        
        #第二行
        self.lbl3 = QLabel("P501：",self.datas)
        self.glo.addWidget(self.lbl3,1,0,1,1)
        self.edit3 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit3,1,1,1,1)
        
        
        self.lbl4 = QLabel("S501：",self.datas)
        self.glo.addWidget(self.lbl4,1,2,1,1)
        self.edit4 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit4,1,3,1,1)
        
        #第三行
        self.lbl5 = QLabel("S502：",self.datas)
        self.glo.addWidget(self.lbl5,2,0,1,1)
        self.edit5 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit5,2,1,1,1)
        
        self.lbl6 = QLabel("B501：",self.datas)
        self.glo.addWidget(self.lbl6,2,2,1,1)
        self.edit6 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit6,2,3,1,1)
        
        #第四行
        self.lbl7 = QLabel("M501：",self.datas)
        self.glo.addWidget(self.lbl7,3,0,1,1)
        self.edit7 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit7,3,1,1,1)
        
        self.lbl8 = QLabel("M502：",self.datas)
        self.glo.addWidget(self.lbl8,3,2,1,1)
        self.edit8 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit8,3,3,1,1)
        
        #第五行
        self.lbl9 = QLabel("O501：",self.datas)
        self.glo.addWidget(self.lbl9,4,0,1,1)
        self.edit9 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit9,4,1,1,1)
        
        
        self.lbl10 = QLabel("R501：",self.datas)
        self.glo.addWidget(self.lbl10,4,2,1,1)
        self.edit10 = QLineEdit(self.datas)
        self.glo.addWidget(self.edit10,4,3,1,1)
        
        InitD = np.array([0.892786538,1.865865385,1.176634615,6.04174565,9.971767726,8.043098836,193.5499117,12.02473812,0.741009028,520.5020252])
        
        self.edit1.setText(str(InitD[0]))
        self.edit2.setText(str(InitD[1]))
        self.edit3.setText(str(InitD[2]))
        self.edit4.setText(str(InitD[3]))
        self.edit5.setText(str(InitD[4]))
        self.edit6.setText(str(InitD[5]))
        self.edit7.setText(str(InitD[6]))
        self.edit8.setText(str(InitD[7]))
        self.edit9.setText(str(InitD[8]))
        self.edit10.setText(str(InitD[9]))
        
        
        self.edit1.setValidator(self.pDoubleValidator)
        self.edit2.setValidator(self.pDoubleValidator)
        self.edit3.setValidator(self.pDoubleValidator)
        self.edit4.setValidator(self.pDoubleValidator)
        self.edit5.setValidator(self.pDoubleValidator)
        self.edit6.setValidator(self.pDoubleValidator)
        self.edit7.setValidator(self.pDoubleValidator)
        self.edit8.setValidator(self.pDoubleValidator)
        self.edit9.setValidator(self.pDoubleValidator)
        self.edit10.setValidator(self.pDoubleValidator)
        
        self.returnbtn = QPushButton("返回")
        self.returnbtn.clicked.connect(self.processReturn)
        self.glo.addWidget(self.returnbtn,5,3,1,1)
        
        self.startbtn = QPushButton("模拟")
        self.startbtn.clicked.connect(self.startTest)
        self.glo.addWidget(self.startbtn,5,1,1,1)
        
        self.vlo.addWidget(self.datas)
        
       
        self.setCentralWidget(self.centralwidget)
        self.setWindowTitle("碳纤维仿真系统")
        self.setWindowIcon(QIcon('./images/winicon.jpg'))
        
    def startTest(self):
     
        self.resultData[0] = float(self.edit1.text())
        self.resultData[1] = float(self.edit2.text())
        self.resultData[2] = float(self.edit3.text())
        self.resultData[3] = float(self.edit4.text())
        self.resultData[4] = float(self.edit5.text())
        self.resultData[5] = float(self.edit6.text())
        self.resultData[6] = float(self.edit7.text())
        self.resultData[7] = float(self.edit8.text())
        self.resultData[8] = float(self.edit9.text())
        self.resultData[9] = float(self.edit10.text())
        
        nt.Nsga2Append(self.resultData)
        
        self.testEnd = processEnd()
        self.testEnd.show()
        self.close()
        
       
    def processReturn(self):
        self.startui = StartUI()
        self.startui.setObjectName("MainWindow")
        self.startui.setStyleSheet("#MainWindow{border-image:url(images/bg.png);}")
        self.startui.show()
        self.close()

        
class StartUI(QMainWindow):
    def __init__(self, parent=None):
        super(StartUI, self).__init__(parent)
        
        self.setFixedSize(400,250)
        self.centralwidget = QWidget()
        
        global currentTime
        time = QDateTime.currentDateTime()
        currentTime = time.toString("yyyy-MM-dd hh:mm:ss")
    
        
        self.btn1 = QPushButton(self.centralwidget)
        self.btn1.setText("正向MLP")
        self.btn1.move(70,210)
        self.btn1.clicked.connect(lambda:self.openUI(self.btn1))
        #self.layout2.addWidget(self.btn1)
        
        self.btn2 = QPushButton(self.centralwidget)
        self.btn2.setText("正向LSTM")
        self.btn2.move(190,210)
        self.btn2.clicked.connect(lambda:self.openUI(self.btn2))
        #self.layout2.addWidget(self.btn2)
        
        self.btn3 = QPushButton(self.centralwidget)
        self.btn3.setText("逆向NSGA2")
        self.btn3.move(310,210)
        self.btn3.clicked.connect(lambda:self.openUI(self.btn3))
        #self.layout2.addWidget(self.btn3)
        
        self.btn4 = QPushButton(self.centralwidget)
        self.btn4.setText("关于")
        self.btn4.move(310,140)
        self.btn4.clicked.connect(lambda:self.openUI(self.btn4))
        
        self.btn1.setToolTip("模拟生产过程碳纤维")
        self.btn2.setToolTip("逆向推导原丝参数")
        self.btn3.setToolTip("训练碳纤维模型")
        self.btn4.setToolTip("介绍程序的功能及使用要求")
        
        self.setCentralWidget(self.centralwidget)
        self.setWindowTitle("碳纤维仿真系统")
        self.setWindowIcon(QIcon('./images/winicon.jpg'))
        #self.setWindowFlags(Qt.FramelessWindowHint)
        
    def openUI(self,btn):
        if btn.text() == "正向MLP":#第一个图标
     
            self.inputstartui = InputStartUI(0)
            self.inputstartui.show()
            self.close()
        elif btn.text() == "正向LSTM":#第二个图标
            
            self.inputstartui = InputStartUI(1)
            self.inputstartui.show()
            self.close()
        elif btn.text() == "逆向NSGA2":#第三个图标
            
            self.inputendui = InputEndUI()
            self.inputendui.show()
            self.close()
        elif btn.text() == "关于":
            
            self.aboutui = aboutUI()
            self.aboutui.show()
            self.close()
    def processtrigger(self,q):
        if q.text() == "quit":
          
            self.testdemo = Menu()
            self.testdemo.show()
            self.close()
     
if __name__ == '__main__':
    app = QApplication(sys.argv)
    startui = StartUI()
    startui.setObjectName("MainWindow")
    startui.setStyleSheet("#MainWindow{border-image:url(images/bg.png);}")
    startui.show()
    sys.exit(app.exec_())
    