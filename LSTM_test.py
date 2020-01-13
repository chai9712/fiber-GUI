# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:20:04 2019

@author: Administrator
"""

import numpy as np
import tensorflow as tf

in_num_c=8
in_num=42
out_num=10
batch_size = 3
MaxV=np.array([15.5,387.537,27.892,204.312,45,28.3907601,76.3,809.35,50,38.71467286,1252.58,69,38.71467286,
               1426.96,79,38.71467286,1537.95,79,38.71467286,1966.04,70,38.71467286,1966.04,70,27.65333776,
               1966.04,70,19.75238411,1966.04,90,13.16825608,1966.04,28,6.584128038,1942.26,180,17.55926251,
               1966.04,0.5,8.547008547,6405.48,0.38,3.418803419,6405.48,25,2.564102564,6405.48,140,8.779631255,
               5.48,2.0511,2.88,1.18,8.810035714,16.4,9.11,241,20.1,1.485756736,936.9331634])
inputV=[]#初始8个参数和后14个炉子的参数
M_y=np.array([0.0 for i in range(out_num)])

def LstmAppend(N,x):
    global inputV
    for i in range(len(x)):
        inputV.append(x[i])
        
def LstmTest(graph,saver,sess):
    with sess.as_default():
         with graph.as_default():                
            graph_lstm = tf.get_default_graph()
            
            test_xc=[]
            test_x=[]
            test_y=[]
            p_c=[]
            global inputV,MaxV
            #print(inputV)
            #传入数据
            count=0
            for i in range(in_num_c):
                p_c.append(inputV[i]/MaxV[count])
                count=count+1
            for i in range(3):
                test_xc.append(p_c)
                
            p=[]
            for i in range(in_num):
                p.append(inputV[i+in_num_c]/MaxV[count])
                count=count+1
            for i in range(3):
                test_x.append(p)
                
            q=[]
            for i in range(out_num):
                q.append(M_y[i])
            for i in range(3):
                test_y.append(q)
            
            feed_dict = {"XC:0": test_xc,"X:0": test_x, "Y:0": test_y,"K:0":0.8}
            prediction = graph_lstm.get_tensor_by_name("prediction:0")
            prediction_value=sess.run(prediction, feed_dict=feed_dict).tolist()
            print("Finished!")       
            inputV=[]
            print(prediction_value)
            
            f=np.array([0.0 for i in range(out_num)])
            for k in range(len(prediction_value)):
                for i in range(out_num):
                    f[i]=f[i]+prediction_value[k][i]/3*MaxV[in_num_c+in_num+i]
            return f       