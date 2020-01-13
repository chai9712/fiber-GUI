# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:49:45 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd 

lr = 0.001
batch_size = 4
total_num=104
train_num=93
test_num=total_num-train_num
num=int(train_num/batch_size)
in_num=50
mid_num=20
out_num=10

def getallData(path):
    sworkbook = xlrd.open_workbook(path)
    sheet = sworkbook.sheet_by_index(0)
    rows = sheet.nrows
    cols = len(sheet.row(0))
    data = []
    for i in range(2,rows):
        temp  = sheet.row_values(i,2,cols)
        data.append(temp)
    return data

def getcolData(data,index,end):
    a = len(data)
    t = []
    for i in range(a):
        p = []
        for j in range(index,end):
            p.append(data[i][j])
        t.append(p)
    return t

def normalize(data):
    data2 = np.array(np.copy(data))
    data2 = data2.transpose()

    r = len(data2)
    c = len(data2[0])
    for i in range(r):
        maxV = np.max(data2[i])
        #minV = np.min(data2[i])
        for j in range(c):
            data2[i][j] = (data2[i][j])/(maxV)
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
                #print(i)
                p.append(data[i][j])
            temp.append(p)
    return temp

def MSE(y,prediction):
    length=len(y)
    s=0
    for i in range(length):
        s=s+(y[i]-prediction[i])*(y[i]-prediction[i])
    MSE=s/length
    return MSE

Data_xy=getallData('纺丝参数_输入输出_End.xlsx')#纺丝参数-输入输出
data_xy=np.array(normalize(Data_xy))#归一化处理
np.random.shuffle(data_xy)#打乱顺序，分训练集和测试集
train_xy=data_xy[0:93,:]
test_xy=data_xy[93:104,:]

train_x=train_xy[:,0:in_num]
train_y=train_xy[:,in_num:in_num+out_num]
test_x=test_xy[:,0:in_num]
test_y=test_xy[:,in_num:in_num+out_num]

xc1 = tf.placeholder(tf.float32, [None, in_num], name='X')
y = tf.placeholder(tf.float32, [None, out_num], name='Y')

h = add_layer(xc1, in_num, mid_num, activation_function=tf.nn.relu)#tf.tanh
prediction = add_layer(h, mid_num, out_num, activation_function=tf.nn.relu)#tf.nn.relu
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
    for i in range(50000):
        batch_xc1=batch_input(train_x,batch_size,i,in_num)
        batch_y=batch_input(train_y,batch_size,i,out_num)
        sess.run(opt, feed_dict={xc1: batch_xc1,y: batch_y})
        if (i+1)%(int(train_num/batch_size)+1)== 0:
            np.random.shuffle(train_xy)#打乱顺序，分训练集和测试集
            train_x=train_xy[:,0:in_num]
            train_y=train_xy[:,in_num:in_num+out_num]
        if (i+1)%250 == 0:
            train_loss = sess.run(loss, feed_dict={xc1: batch_xc1,y: batch_y})
            train_LOSS.append(train_loss)
            
            batch_testX=batch_input(test_x,test_num,0,in_num)
            batch_testY=batch_input(test_y,test_num,0,out_num)
            tt=sess.run(loss, feed_dict={xc1: batch_testX,y: batch_testY})
            test_LOSS.append(tt)
#            if(tt<curr):
#                curr=tt
#                count=0
#            else:
#                count=count+1
#            if(count==30):
#                break
            
            
            print ("Iter=%d, loss=%g" % ((i+1), train_loss))
            if train_loss<0.2 and train_loss>0.1:
                lr=0.0005
            if train_loss<0.1:
                lr=0.0002
            if train_loss<0.05:
                lr=0.0001
                
    xx=np.arange(len(test_LOSS))   
    plt.plot(xx, train_LOSS,color='red')  
    plt.plot(xx, test_LOSS,color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('Training Loss--Iterations')
    plt.show()  
    
    xt=np.arange(total_num)
    for i in range(train_num):
        train_xc1=batch_input(train_x,1,i,in_num)
        train_y1=batch_input(train_y,1,i,out_num)
        prediction_value=prediction_value+sess.run(prediction, feed_dict={xc1: train_xc1,y: train_y1}).tolist()
        
    for i in range(test_num):
        batch_testX=batch_input(test_x,1,i,in_num)
        batch_testY=batch_input(test_y,1,i,out_num)
        prediction_value=prediction_value+sess.run(prediction, feed_dict={xc1: batch_testX,y: batch_testY}).tolist()
        
    for k in range(out_num):
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
    for n in range(out_num):
        SSE=SSE+MSE(temp_y1[:,n],train_Y[:,n])
        SST=SST+np.var(temp_y1[:,n])
    eval_R=1-SSE/SST    
    print ("Train Effect=%f" % eval_R)
    
   
    for n in range(out_num):
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
            for m in range(out_num):
                if abs(prediction_value[n][m]-train_y[n][m])<train_y[n][m]*0.15:
                    temp=temp+1
            sum_c1=sum_c1+temp
            if temp==10:
                sum_p1=sum_p1+1
        else:
            for m in range(out_num):
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
            for m in range(out_num):
                if abs(prediction_value[n][m]-train_y[n][m])<0.1:
                    temp=temp+1
            Sum_c1=Sum_c1+temp
            if temp==10:
                Sum_p1=Sum_p1+1
        else:
            for m in range(out_num):
                if abs(prediction_value[n][m]-test_y[n-train_num][m])<0.1:
                    temp=temp+1
            Sum_c2=Sum_c2+temp
            if temp==10:
                Sum_p2=Sum_p2+1
        Sum_s.append(temp)
        
    print ("train accuracy=%f" % (Sum_p1/train_num))
    print ("train accuracy=%f" % (Sum_c1/10/train_num))
    if(test_num>0):
        print ("test accuracy=%f" % (Sum_p2/test_num))
        print ("test accuracy=%f" % (Sum_c2/10/test_num))
    
    plt.plot(xt, Sum_s,color='red')  
    plt.show() 
    
    saver.save(sess, "Model_MLP/UniMLP_50_10")#保存模型
