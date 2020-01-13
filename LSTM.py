# -*- coding: utf-8 -*-
"""
Created on Mon May  6 23:32:20 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import xlrd

lr = 0.001
#MLP层
in_num=8
mid_num=3+7
out_num=3
#LSTM层
input_size = 3    # 每个时刻的输入特征是4维的
timestep_size = 15    #总共输入15次
#num_units=32        #LSTM隐含层节点个数
hidden_size = 6+6    #每个隐含层的节点数
output_num = 10      #输出结果的个数

batch_size = 3
total_num=104
train_num=93
test_num=total_num-train_num

def proTemp(tempData):
    result = []
    num = len(tempData)
    for i in range(num):
        if type(tempData[i])==type(float(2.2)):
            result.append(tempData[i])
    return result

def getallData(path):
    sworkbook = xlrd.open_workbook(path)
    sheet = sworkbook.sheet_by_index(0)
    rows = sheet.nrows
    cols = len(sheet.row(0))
    data = []
    for i in range(2,rows):
        temp  = sheet.row_values(i,2,cols)
        data.append(proTemp(temp))
    return data

def batch_input(data,batch_size,num,cont_size):
    temp=[]
    index=(num*batch_size)%train_num
    if(index+batch_size>train_num):
        for i in range(index,train_num):
            p=[]
            for j in range(0,cont_size):
                p.append(data[i][j])
            temp.append(p)
        for i in range(0,index+batch_size-train_num):
            p=[]
            for j in range(0,cont_size):
                p.append(data[i][j])
            temp.append(p)
    else:
        for i in range(index,index+batch_size):
            p=[]
            for j in range(0,cont_size):
                p.append(data[i][j])
            temp.append(p)
    return temp

def batch_input2(data,batch_size,num,cont_size):
    temp=[]
    index=(num*batch_size)%test_num
    if(index+batch_size>test_num):
        for i in range(index,test_num):
            p=[]
            for j in range(0,cont_size):
                p.append(data[i][j])
            temp.append(p)
        for i in range(0,index+batch_size-test_num):
            p=[]
            for j in range(0,cont_size):
                p.append(data[i][j])
            temp.append(p)
    else:
        for i in range(index,index+batch_size):
            p=[]
            for j in range(0,cont_size):
                p.append(data[i][j])
            temp.append(p)
    return temp

def normalize(data):
    data2 = np.array(np.copy(data))
    data2 = data2.transpose()

    r = len(data2)
    c = len(data2[0])
    for i in range(r):
        maxV = np.max(data2[i])
        for j in range(c):
            data2[i][j] = data2[i][j]/maxV
    
    return data2.transpose().tolist()

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights1 = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.01), dtype=tf.float32)
    Biases1 = tf.Variable(tf.zeros([1, out_size]) + 0.01)
    Wx_plus_b = tf.matmul(inputs, Weights1) + Biases1
    
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def unit_lstm():
    # 定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    #添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell

def MSE(y,prediction):
    length=len(y)
    s=0
    for i in range(length):
        s=s+(y[i]-prediction[i])*(y[i]-prediction[i])
    MSE=s/length
    return MSE

Data_xy=getallData('纺丝参数_输入输出_End.xlsx')
data_xy=np.array(normalize(Data_xy))#归一化处理
np.random.shuffle(data_xy)#打乱顺序，分训练集和测试集

train_xy=data_xy[0:93,:]
test_xy=data_xy[93:104,:]

train_c=train_xy[:,0:8]
train_x=train_xy[:,8:50]
train_y=train_xy[:,50:60]

test_c=test_xy[:,0:8]
test_x=test_xy[:,8:50]
test_y=test_xy[:,50:60]

xc = tf.placeholder(tf.float32, [None, in_num], name='XC')
h = add_layer(xc, in_num, mid_num, activation_function=tf.nn.relu)#tf.tanh
prediction_c = add_layer(h, mid_num, out_num, activation_function=None)#tf.nn.relu


x = tf.placeholder(tf.float32, [None, (timestep_size-1)*input_size], name='X')
y = tf.placeholder(tf.float32, [None, output_num], name='Y')
keep_prob = tf.placeholder(tf.float32, name='K')

xl=tf.concat([prediction_c,x],1)
x_input=tf.reshape(xl, [-1,15, 3])


#调用 MultiRNNCell 来实现多层 LSTM
mlstm_cell = rnn.MultiRNNCell([unit_lstm() for i in range(1)], state_is_tuple=True)
#用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x_input, initial_state=init_state, time_major=False)
h_state = outputs[:, -1, :] 

weights = tf.Variable(tf.truncated_normal([hidden_size, output_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[output_num]), dtype=tf.float32)
prediction=tf.matmul(h_state, weights) + bias
predition_abs=tf.abs( prediction, name='prediction') 
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction),reduction_indices=[1]))
opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

saver = tf.train.Saver()  
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_LOSS=[]
    test_LOSS=[]
    prediction_value=[]
    curr=1
    count=0
    for i in range(10000):
        batch_xc= batch_input(train_c,batch_size,i,8)
        batch_x = batch_input(train_x,batch_size,i,42)
        batch_y = batch_input(train_y,batch_size,i,10)
        sess.run(opt, feed_dict={xc: batch_xc,x: batch_x, y: batch_y, keep_prob: 0.8})       
        
        if (i+1)%(int(train_num/batch_size)+1)== 0:
            np.random.shuffle(train_xy)#打乱顺序，分训练集和测试集
            train_c=train_xy[:,0:8]
            train_x=train_xy[:,8:50]
            train_y=train_xy[:,50:60]
            
        if (i+1)%50 == 0:
            train_loss = sess.run(loss, feed_dict={xc: batch_xc,x: batch_x, y: batch_y, keep_prob: 0.8})
            train_LOSS.append(train_loss)
            print ("Iter=%d, loss=%g" % ((i+1), train_loss))
            if train_loss<0.3 and train_loss>0.15:
                lr=0.0005
            if train_loss<0.15:
                lr=0.0002
            if train_loss<0.05:
                lr=0.0001
            
            tt=0
            for k in range(int(test_num/batch_size)+1):
                batch_testC=batch_input2(test_c,batch_size,k,8)
                batch_testX=batch_input2(test_x,batch_size,k,42)
                batch_testY=batch_input2(test_y,batch_size,k,10)
                tt=tt+sess.run(loss, feed_dict={xc: batch_testC,x: batch_testX, y: batch_testY, keep_prob: 0.8})
            test_LOSS.append(tt/(int(test_num/batch_size)+1))
#            if(tt<curr):
#                curr=tt
#                count=0
#            else:
#                count=count+1
#            if(count==30):
#                break
        
        
    xx=np.arange(len(test_LOSS))   
    plt.plot(xx, train_LOSS,color='red')  
    plt.plot(xx, test_LOSS,color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('Training Loss--Iterations')
    plt.show()  
    
    res=[]
    xt=np.arange(total_num)
    for i in range(int(train_num/batch_size)+1):
        train_c1=batch_input(train_c,batch_size,i,8)
        train_x1=batch_input(train_x,batch_size,i,42)
        train_y1=batch_input(train_y,batch_size,i,10)
        res=res+sess.run(prediction, feed_dict={xc: train_c1,x: train_x1, y: train_y1, keep_prob: 0.8}).tolist()
    count=0
    for i in range(len(res)):
        if(count<train_num):
            prediction_value.append(res[i])
            count=count+1
    res=[]
    for i in range(int(test_num/batch_size)+1):
        test_c1=batch_input2(test_c,batch_size,i,8)
        test_x1=batch_input2(test_x,batch_size,i,42)
        test_y1=batch_input2(test_y,batch_size,i,10)
        res=res+sess.run(prediction, feed_dict={xc: test_c1,x: test_x1, y: test_y1, keep_prob: 0.8}).tolist()
        
    for i in range(len(res)):
        if(count<total_num):
            prediction_value.append(res[i])
            count=count+1
                
    for k in range(output_num):
        p=[]
        t=[]
        for n in range(total_num):
            p.append(prediction_value[n][k])
            if n<train_num:
                t.append(train_y[n][k])
            else:
                t.append(test_y[n-train_num][k])
        plt.plot(xt, p,color='red')  
        plt.plot(xt, t,color='green')
        plt.xlabel('NO.')
        plt.ylabel('Value')
        plt.title('Accuracy')
        plt.show()  

    eval_R=0
    SSE=0  
    SST=0
    temp_y1=np.array(train_y)
    temp_y2=np.array(test_y)
    temp_p=np.array(prediction_value)
    train_Y=temp_p[0:93,:]
    test_Y=temp_p[93:104,:]
    for n in range(output_num):
        SSE=SSE+MSE(temp_y1[:,n],train_Y[:,n])
        SST=SST+np.var(temp_y1[:,n])
    eval_R=1-SSE/SST    
    print ("Train Effect=%f" % eval_R)
    
   
    for n in range(output_num):
        SSE=SSE+MSE(temp_y2[:,n],test_Y[:,n])
        SST=SST+np.var(temp_y2[:,n])
    eval_R=1-SSE/SST    
    print ("Test Effect=%f" % eval_R)
    
    sum_s=[]
    sum_c1=0
    sum_p1=0
    sum_c2=0
    sum_p2=0
    
    for n in range(total_num):
        temp=0        
        if n<train_num:
            for m in range(output_num):
                if abs(prediction_value[n][m]-train_y[n][m])<train_y[n][m]*0.15:
                    temp=temp+1
            sum_c1=sum_c1+temp
            if temp==10:
                sum_p1=sum_p1+1
        else:
            for m in range(output_num):
                if abs(prediction_value[n][m]-test_y[n-train_num][m])<test_y[n-train_num][m]*0.1:
                    temp=temp+1
            sum_c2=sum_c2+temp
            if temp==10:
                sum_p2=sum_p2+1
        sum_s.append(temp)
        
    print ("train accuracy=%f" % (sum_p1/train_num))
    print ("train accuracy=%f" % (sum_c1/10/train_num))
    if(test_num>0):
        print ("test accuracy=%f" % (sum_p2/test_num))
        print ("test accuracy=%f" % (sum_c2/10/test_num))
    
    plt.plot(xt, sum_s,color='red')  
    plt.show() 
    
    
    Sum_s=[]
    Sum_c1=0
    Sum_p1=0
    Sum_c2=0
    Sum_p2=0
    
    for n in range(total_num):
        temp=0
        
        if n<train_num:
            for m in range(output_num):
                if abs(prediction_value[n][m]-train_y[n][m])<0.1:
                    temp=temp+1
            Sum_c1=Sum_c1+temp
            if temp==10:
                Sum_p1=Sum_p1+1
        else:
            for m in range(output_num):
                if abs(prediction_value[n][m]-test_y[n-train_num][m])<0.1:
                    temp=temp+1
            Sum_c2=Sum_c2+temp
            if temp==10:
                Sum_p2=Sum_p2+1
        Sum_s.append(temp)
        
    print ("train accuracy=%f" % (Sum_p1/train_num))
    print ("train accuracy=%f" % (Sum_c1/output_num/train_num))
    if(test_num>0):
        print ("test accuracy=%f" % (Sum_p2/test_num))
        print ("test accuracy=%f" % (Sum_c2/output_num/test_num))
    
    plt.plot(xt, Sum_s,color='red')  
    plt.show() 
    
    saver.save(sess, "Model_LSTM/LSTM_8_42_10") 
    