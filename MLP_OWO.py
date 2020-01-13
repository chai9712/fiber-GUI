# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:10:07 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import xlrd 
import math
import random

lr = 0.001
batch_size = 4
total_num=104
train_num=93
test_num=total_num-train_num
num=int(train_num/batch_size)
in_num=50
Nh=20
out_num=10
hmean = 0.1
hvar = 0.1

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
    
def variance(N,batch,batch_size,xm,xv):
    for i in range(N):
        for j in range(batch_size):
            xm[i] += batch[j][i]
    for i in range(N): 
        xm[i] = xm[i] / batch_size
    for i in range(N):
        for j in range(batch_size):        
            xv[i] += (batch[j][i]-xm[i])*(batch[j][i]-xm[i])
    for i in range(N):       
        xv[i] = xv[i] / batch_size
        
def slete(mu,sigma,k):
    mu = 0.1
    sigma = 0.5
    np.random.seed(1)
    s1 = list(np.random.normal(mu, sigma, k*k))
    s = random.sample(s1,k)
    return s

def wihnorm(wih,N,Nh):
    mm = [0.0 for i in range(Nh)]
    vv = [0.0 for i in range(Nh)]
    for i in range(Nh):
        for k in range(N):
            mm[i] += wih[i][k]
    for i in range(Nh):
        mm[i] = mm[i] / N
    for i in range(Nh):
        for k in range(N):
            vv[i] += (wih[i][k]-mm[i])*(wih[i][k]-mm[i])
    for i in range(Nh):
        vv[i] = math.sqrt(vv[i] / N)
    for i in range(Nh):
        for k in range(N):
            wih[i][k] = (wih[i][k] - mm[i])/vv[i]
            
def hnet(k,N,Nh,wih,th,batch,u):
    h1 = 0.0
    for i in range(N):
        h1 += batch[u][i]*wih[k][i]
    h1 = h1+th[k]
    return h1

def hact(h1):
    h2 = 1 / (1 + math.exp(-h1))
    return h2

def Schmit(R,C,Nu,M,Et,w):
    NLin = 0
    gmin = 0.000001
    g = sumck = 0.0
    MSE = 0.0
    a = [[0.0]*Nu for i in range(Nu)]
    b = [0.0 for i in range(Nu)] 
    c = [0.0 for i in range(Nu)]   
    E = [0.0 for i in range(M)] 
    Wt = [[0.0]*Nu for i in range(M)]
    for i in range(M):
        E[i] = Et[i]
    gmin = gmin*gmin
    g = (R[0][0])
    
    if g<gmin:
        a[0][0] = 0.0
        NLin = NLin + 1
    else:
        g = math.sqrt(g)
        a[0][0] = 1 / g
    c[0] = a[0][0]*R[0][1]
    b[0] = -c[0]*a[0][0]
    b[1] = 1
    g = (R[1][1]-c[0]*c[0])
    
    if g<gmin:
        a[1][0] = 0.0
        a[1][1] = 0.0
        NLin = NLin + 1
    else:
        g = math.sqrt(g)
        a[1][0] = b[0]*(1/g)
        a[1][1] = b[1]*(1/g)
    
    for i in range(2,Nu):
        for j in range(i):
            c[j] = 0
            for k in range(j+1):
                c[j] = c[j]+a[j][k]*R[k][i]
        b[i] = 1
        for j in range(i):
            b[j] = 0
            for k in range(j,i):
                b[j] = b[j]-c[k]*a[k][j]       
        sumck = 0
        for k in range(i):
            sumck += c[k]*c[k]
        g = (R[i][i]-sumck)
        if g<gmin:
            for k in range(i+1):
                a[i][k] = 0
            NLin = NLin + 1
        else:
            g = 1 / math.sqrt(g)
            for k in range(i+1):
                a[i][k] = b[k]*g
                
    for i in range(M):
        for m in range(Nu):
            Wt[i][m] = 0.0
            for k in range(m+1): 
                Wt[i][m] += a[m][k]*C[i][k]
    for i in range(M):
        for k in range(Nu):
            E[i] = E[i] -Wt[i][k]*Wt[i][k]
    for i in range(M):
        MSE = MSE + abs(E[i])
         
    for i in range(M):
        for k in range(Nu):
            w[i][k] = 0.0
            for m in range(k,Nu):
                w[i][k] = w[i][k] + a[m][k]*Wt[i][m]
    return MSE 

def backprop(N,M,Nh,Nu,batch,xa,w,wih,th,dwih,dth,h,who):
    dpo = np.array([0.0 for i in range(M)])
    dph = np.array([0.0 for i in range(Nh)])    
    y = np.array([0.0 for i in range(M)])

    for k in range(Nh):
            dth[k] = 0.0
    for k in range(Nh):
        for i in range(N):
            dwih[k][i] = 0.0
#    print(batch)
    for j in range(batch_size):  
        for k in range(M):
            y[k] = 0.0
        for i in range(N):
            xa[i] = batch[j][i]
        for i in range(Nh):
            h[i] = hnet(i,N,Nh,wih,th,batch,j) 
        for i in range(N+1,Nu):
            xa[i] = hact(h[i-N-1])
    
        xa[N] = 1
#        print(xa)
        for k in range(M):
            for u in range(Nu):
                y[k] += xa[u]*w[k][u]
            dpo[k] = 2.0*(batch[j][N+k]-y[k])
        print(dpo)
        for k in range(Nh):
            dph[k] = 0.0
        for n in range(Nh):
            for k in range(M):
                dph[n] +=who[k][n]*dpo[k]
            #print(dph[n])
            dph[n] *= hact(hnet(n,N,Nh,wih,th,batch,j))*(1-hact(hnet(n,N,Nh,wih,th,batch,j)))
            #print(dph[n])
            
        for n in range(Nh):
            dth[n] = dth[n]+dph[n]*1
            for i in range(N):
               dwih[n][i] += dph[n]*batch[j][i]
        
    for j in range(Nh):
        dth[j] /= batch_size
        for i in range(N):
            dwih[j][i] /= batch_size
    
        
def olffunc(N,M,Nh,Nu,batch,xa,wih,dwih,th,w,dth):
    fder = [0.0 for i in range(Nh)] 
    d1 = 0.0
    d2 = 0.0
    
    for j in range(batch_size):  
        for i in range(N):
            xa[i] = batch[j][i]
        xa[N] = 1.0
        t1 = [0 for i in range(Nh)]  
        n = [0 for i in range(Nh)] 
        OO = [0 for i in range(Nh)] 
        for k in range(Nh):
            for nn in range(N):
                n[k] = n[k] + wih[k][nn] * xa[nn]         
            for nn in range(N):
                t1[k] = t1[k] + dwih[k][nn]*xa[nn]
            t1[k] = t1[k] + dth[k]
            n[k] = n[k] +th[k]
            OO[k] = hact(n[k])
            fder[k] = OO[k]*(1-OO[k])
        t2 = [0 for i in range(M)] 
        yy = [0 for i in range(M)]
        for i in range(M):
            for n1 in range(N):
                yy[i] = yy[i] + w[i][n1]*xa[n1]
            yy[i] = yy[i] + w[i][N]
            for k in range(Nh):
                yy[i] = yy[i] + w[i][N+1+k] * OO[k]
                t2[i] = t2[i] + w[i][N+1+k] * fder[k] * t1[k]
            d1 = d1+(batch[j][i]-yy[i])*t2[i]
            d2 = d2+t2[i]*t2[i]
#    print(d1)
#    print(d2)
#    print("A")
    Z = d1 / d2
    return Z

def  rcfunc(N,M,Nh,Nu,batch,xa,w,wih,h,th,R,C):
     for j in range(Nu):     
        for i in range(Nu):
            R[j][i] = 0.0
     for j in range(M):     
        for i in range(Nu):
            C[j][i] = 0.0 
     for j in range(M):     
        for i in range(Nu):
            w[j][i] = 0.0 
            
     for j in range(batch_size):       
        for i in range(N):
            xa[i] = batch[j][i]
        for i in range(Nh):
            h[i] = hnet(i,N,Nh,wih,th,batch,j) 
#       
        for i in range(N+1,Nu):
            xa[i] = hact(h[i-N-1])
        xa[N] = 1 
        for i in range(Nu):
            for k in range(Nu):
                R[i][k] += xa[i]*xa[k]
        for i in range(M):
            for k in range(Nu):
                C[i][k] += batch[j][N+i]*xa[k]
     for i in range(Nu):
        for j in range(Nu):
            R[i][j] = R[i][j] / batch_size
     for i in range(M):
        for j in range(Nu):
            C[i][j] = C[i][j] / batch_size
            

    
data_xy=np.array(getallData('纺丝参数_输入输出_End.xlsx'))#纺丝参数-输入输出
total_row = len(data_xy)
total_col=len(data_xy[0])
dataM = np.array([0.0 for i in range(total_col)])
dataV = np.array([0.0 for i in range(total_col)])      
variance(total_col,data_xy,total_row,dataM,dataV) 
#0均值归一化数据
for j in range(total_row):
    for i in range(total_col):
        if dataV[i]>1.0e-8:
            data_xy[j][i] = (data_xy[j][i]-dataM[i])/math.sqrt(dataV[i])
        else:
            data_xy[j][i] = 0
            
N=50
M=10
temp=data_xy
xym=dataM
xyv=dataV
Nu = N+Nh+1
wih =np.array([[0.0]*N for i in range(Nh)])
th = np.array([0.0 for i in range(Nh)])
wio = np.array([[0.0]*N for i in range(M)])
who = np.array([[0.0]*Nh for i in range(M)])
w = np.array([[0.0]*Nu for i in range(M)])
h = np.array([0.0 for i in range(Nh)])
hv = np.array([0.0 for i in range(Nh)]) 
hm = np.array([0.0 for i in range(Nh)]) 
Et = np.array([0.0 for i in range(M)])
R = np.array([[0.0]*Nu for i in range(Nu)])
C = np.array([[0.0]*Nu for i in range(M)])
xa = np.array([0.0 for i in range(Nu)])
dth = np.array([0.0 for i in range(Nh)])
dwih = np.array([[0.0]*N for i in range(Nh)])
#初始化输入层到隐藏层的权重和偏置
for j in range(Nh):
    wih[j] = slete(1.0,0.5,N)
    for i in range(N):
        if xyv[i]>1.0e-8:
            wih[j][i] = wih[j][i]/math.sqrt(xyv[i])
        else:
            wih[j][i] = 0
    th[j] = slete(1.0,0.5,1)[0]
#归一化权重wih
wihnorm(wih,N,Nh)
#初始化输入层到输出层的权重和偏置
for k in range(M):
    for i in range(N):
        wio[k][i] = 0.0
 #初始化隐藏层到输出层的权重和偏置
for k in range(M):
    for j in range(Nh):
        who[k][j] = 0.0
#初始化输入层+隐含层+1到输出层的权重
for k in range(M):
    for u in range(Nu):
        w[k][u] = 0.0        
#初始化隐含层
for i in range(Nh):
    h[i] = 0.0
    hv[i] = 0.0
for k in range(M):
    Et[k] = 0

for u in range(total_row):
    for j in range(Nh):
        h[j] += hnet(j,N,Nh,wih,th,temp,u) 
for j in range(Nh):
    hm[j] = h[j] / total_row
for u in range(total_row):    
    for j in range(Nh):
        h[j] = hnet(j,N,Nh,wih,th,temp,u)
        hv[j] += (h[j]-hm[j])*(h[j]-hm[j])
for j in range(Nh):
    hv[j] /= total_row
    hv[j] = math.sqrt(hv[j]) 

for j in range(Nh):
    for i in range(N):
        wih[j][i] = wih[j][i]/hv[j]
    th[j] = th[j] / hv[j]
    th[j] = th[j]-(hm[j]*hvar/hv[j]) +hmean
    
for j in range(total_row):       
    for i in range(N):
        xa[i] = temp[j][i]
    for i in range(Nh):
        h[i] = hnet(i,N,Nh,wih,th,temp,j) 
    for i in range(N+1,Nu):
        xa[i] = hact(h[i-N-1])
    xa[N] = 1
    for i in range(Nu):
        for k in range(Nu):
            R[i][k] += xa[i]*xa[k]
    for i in range(M):
        for k in range(Nu):
            C[i][k] += temp[j][N+i]*xa[k]
for i in range(Nu):
    for j in range(Nu):
        R[i][j] = R[i][j] / total_row
for i in range(M):
    for j in range(Nu):
        C[i][j] = C[i][j] / total_row
        
#####################################################
#迭代更新  
E = Schmit(R,C,Nu,M,Et,w)#更新w
Nit=1
Z1 = 0.0    
LOSS=[]
for it in range(Nit):
    if (it+1)%(int(train_num/batch_size)+1)== 0:
        np.random.shuffle(temp)#打乱顺序，分训练集和测试集
    #It=0
    batch=batch_input(temp,batch_size,it,in_num+out_num)

    for j in range(batch_size):    
        for k in range(M):
            Et[k] = Et[k]+batch[j][N+k]*batch[j][N+k]
    for k in range(M):
        Et[k] /= batch_size
    for k in range(M):
        for j in range(Nh):
            who[k][j] = w[k][j + N + 1]
        
    #获得dwih,dth
    backprop(N,M,Nh,Nu,batch,xa,w,wih,th,dwih,dth,h,who)
    
    #获得学习率Z
#    Z1=0.001
#    Z1 = olffunc(N,M,Nh,Nu,batch,xa,wih,dwih,th,w,dth)
#    #更新wih，th
#    for j in range(Nh):
#        th[j] +=Z1*(dth[j])
#        for i in range(N):
#            wih[j][i] += Z1*dwih[j][i]
#    #更新R，C
#    rcfunc(N,M,Nh,Nu,batch,xa,w,wih,h,th,R,C)
#    #获得误差，更新矩阵w
#    E = Schmit(R,C,Nu,M,Et,w)
#    if((it+1)%10==0):
#        LOSS.append(E)
#        print("%s %s %s\n"%(it+1,Z1,E))
#        
#xx=np.arange(len(LOSS))
#plt.plot(xx, LOSS,color='red')  
#plt.show() 
#
#













