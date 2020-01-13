# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:07:06 2019

@author: Administrator
"""
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import tensorflow as tf

in_num=50
out_num=10
MaxV=np.array([15.5,387.537,27.892,204.312,45,28.3907601,76.3,809.35,50,38.71467286,1252.58,69,38.71467286,1426.96,79,38.71467286,1537.95	,79,38.71467286,1966.04,70,38.71467286,1966.04,70,27.65333776,1966.04,70,19.75238411,1966.04,90,13.16825608,1966.04,28,6.584128038,1942.26,180,17.55926251,1966.04,0.5,8.547008547,6405.48,0.38,3.418803419,6405.48,25,2.564102564,6405.48,140,8.779631255,6405.48,2.0511,2.88,1.18,8.810035714,16.4,9.11,241,20.1,1.485756736,936.9331634])

Act=np.array([1.0503,0.7,1.17,6.81,14.93,8.38,191,10.33,0.71,490.5570424])
M_y=np.array([0.0 for i in range(out_num)])  

def Nsga2Append(y):
    for i in range(out_num):
        M_y[i]=y[i]/MaxV[in_num+i]

def myfind(x,y):#返回一维数组中指定值的索引
    return [ a for a in range(len(y)) if y[a] == x]

def initialize_variables(N, M, V, min_range, max_range,graph,saver,sess):#N种群数目，M目标数量，V决策数量
    minV = min_range
    maxV = max_range
    K = M + V   #K是数组的总元素个数。为了便于计算，决策变量和目标函数串在一起形成一个数组。  
    #对于交叉和变异，利用目标变量对决策变量进行选择
    f = [[0.0]*K for i in range(N)]#f是一个由种群个体组成的二维数组(N*K)
    for i in range(N):
        for j in range(V):
            f[i][j] = minV[j] + (maxV[j] - minV[j])*random.random()#f(i j)表示的是种群中第i个个体中的第j个决策变量，
                                                         #这行代码为每个个体的所有决策变量在约束条件内随机取值
        temp = evaluate_objective(f[i][0:V], M, V,graph,saver,sess) # M是目标函数数量 V是决策变量个数
        #为了简化计算将对应的目标函数值储存在染色体的V + 1 到 K的位置。
        for p in range(M):
            f[i][V+p]=temp[p]
    return f  

def non_domination_sort_mod(x, M, V): #x为初始种群(N*(V+M))
    N=len(x) #N为矩阵x的行数，也是种群的数量
    F = []
    individual_n = []
    individual_p = []
    x=np.array(x)
    x_hier = [0 for i in range(N)]#用于存储个体染色体中加入分级信息
    x=np.column_stack((x,x_hier))
    
    temp_F=[]
    for i in range(N):
        n = 0#n是个体i被支配的个体数量
        p = []#p是被个体i支配的个体集合
        for j in range(N):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for k in range(M):       #判断个体i和个体j的支配关系
                if (x[i][V + k] < x[j][V + k]):
                    dom_less = dom_less + 1
                elif(x[i][V + k] == x[j][V + k]):
                    dom_equal = dom_equal + 1
                else:
                    dom_more = dom_more + 1

            if (dom_less == 0 and dom_equal != M): # 说明i受j支配，相应的n加1
                n = n + 1
            if (dom_more == 0 and dom_equal != M): # 说明i支配j,把j加入i的支配合集中
                p.append(j)
        individual_n.append(n)
        individual_p.append(p)
 
        if (n == 0): #个体i非支配等级排序最高，属于当前最优解集，相应的染色体中携带代表排序数的信息
            x_hier[i] = 1
            temp_F.append(i)#等级为0的非支配解集
    F.append(temp_F)
    #上面的代码是为了找出等级最高的非支配解集
    
    front = 0
    #下面的代码是为了给其他个体进行分级
    while len(F[front])!=0:
       Q = [] #存放下一个front集合
       for i in range(len(F[front])):#循环当前支配解集中的个体
           if len(individual_p[F[front][i]])!=0:#个体i有自己所支配的解集
    	        for j in range(len(individual_p[F[front][i]])):#循环个体i所支配解集中的个体
    	            individual_n[individual_p[F[front][i]][j]] = individual_n[individual_p[F[front][i]][j]] - 1
                   #...表示的是与下一行代码是相连的， 这里表示个体j的被支配个数减1
    	            if (individual_n[individual_p[F[front][i]][j]]==0):# 如果q是非支配解集，则放入集合Q中
                      x[individual_p[F[front][i]][j]][M + V] =  front + 1#个体染色体中加入分级信息
                      Q.append(individual_p[F[front][i]][j])
#       print(Q)
       front =  front + 1
       F.append(Q)
#    print(F)
    
    sorted_based_on_front=[]
    index_of_fronts=np.argsort(x[:,M + V])#对个体的代表排序等级的列向量进行升序排序 index_of_fronts表示排序后的值对应原来的索引
    for i in range(len(index_of_fronts)):
        sorted_based_on_front.append(x[index_of_fronts[i]])#存放的是x矩阵按照排序等级升序排序后的矩阵x(N*(V+M+1))
    
    current_index = 0
    #计算每个个体的拥挤度
    z=[]
    z_s=[]
    z_count=0

    for front in range(len(F)-1):#减1是因为代码80行这里，F的最后一个元素为空，这样才能跳出循环。一共有length-1个排序等级
        distance = 0
        y = np.array([[0.0]*len(sorted_based_on_front[0]) for i in range(len(F[front]))])#变为数组，方便获取列数据
        #previous_index = current_index + 1
        
        for i in range(len(F[front])):
            for p in range(len(sorted_based_on_front[0])):
                y[i,p]=sorted_based_on_front[current_index + i][p]#y中存放的是排序等级为front的集合矩阵
                    
        current_index = current_index + i #current_index =i
        sorted_based_on_objective = []#存放基于拥挤距离排序的矩阵
        y_dis=np.array([[0.0]*M for i in range(len(F[front]))])#添加距离信息
        y=np.column_stack((y,y_dis)) # len(F[front]) * (V+M+1+M)
        
        for i in range(M):
            index_of_objectives=np.argsort(y[:,V + i])
            #sorted_based_on_objective = sorted(y[:,V + i])#按照目标函数值排序
            sorted_based_on_objective = []
            
            for j in range(len(index_of_objectives)):
                sorted_based_on_objective.append(y[index_of_objectives[j]])#存放按照目标函数值排序后的x矩阵

            f_max = sorted_based_on_objective[len(index_of_objectives)-1][ V + i]#fmax为目标函数最大值 fmin为目标函数最小值
            f_min = sorted_based_on_objective[0] [V + i]
            
            y[index_of_objectives[len(index_of_objectives)-1]][M + V + 1 + i]=10000#对排序后的第一个个体和最后一个个体的距离设为无穷大
            y[index_of_objectives[0]][M + V + 1 + i] = 10000
            
            for j in range(1,len(index_of_objectives) - 1):#循环集合中除了第一个和最后一个的个体
                next_obj  = sorted_based_on_objective[j + 1][V + i]
                previous_obj  = sorted_based_on_objective[j - 1][V + i]
                if (f_max - f_min == 0):
                    y[index_of_objectives[j]][M + V + 1 + i] = 10000
                else:
                    y[index_of_objectives[j]][M + V + 1 + i] =(next_obj - previous_obj)/(f_max - f_min)
             
        
        distance = np.array([0.0 for i in range(len(F[front]))])
        for i in range(M):
            distance=[distance[p]+y[p,M + V + 1 + i] for p in range(len(distance))]
      
        z_count=z_count+len(distance)
        z_s.append(len(distance))
        y[:,M + V + 2] = distance
        y = y[:,0 : M + V + 2]
        z.append(y)
    f = np.array([[0.0]*(M + V + 2) for i in range(z_count)])
#    print(z)
    temp=0
    for i in range(len(z)):
        for j in range(len(z[i])):
            for k in range(len(z[i][0])):
                f[temp][k]=z[i][j][k]
            temp=temp+1
    
    return f
    

def tournament_selection(chromosome, pool_size, tour_size):
    pop = len(chromosome)         #获得种群的个体数量
    variables=len(chromosome[0])  #获得决策变量数量
    rank = variables - 1          #个体向量中排序值所在位置
    distance = variables          #个体向量中拥挤度所在位置
    #竞标赛选择法，每次随机选择两个个体，优先选择排序等级高的个体，如果排序等级一样，优选选择拥挤度大的个体
    f=[]
    for i in range (pool_size):
        candidate = [0 for i in range(tour_size)] 
        for j in range (tour_size):
            candidate[j] = round(pop*random.random())#随机选择参赛个体
            if (candidate[j] == pop):
                candidate[j] = pop-1

            if (j > 1):
                while (candidate[1 : j - 1].count(candidate[j])==0):#防止两个参赛个体是同一个
                    candidate[j] = round(pop*random.random())
                    if (candidate[j] == 0):
                        candidate[j] = 1
        c_obj_rank = np.array([0.0 for i in range(tour_size)])
        c_obj_distance = np.array([0.0 for i in range(tour_size)])
        for j in range(tour_size): # 记录每个参赛者的排序等级 拥挤度
            #print(candidate[j])
            c_obj_rank[j] = chromosome[candidate[j]][rank]
            c_obj_distance[j] = chromosome[candidate[j]][distance-1]
            
        min_candidate=myfind(min(c_obj_rank),c_obj_rank)#选择排序等级较小的参赛者，find返回该参赛者的索引
        if len(min_candidate) != 1:#如果两个参赛者的排序等级相等 则继续比较拥挤度 优先选择拥挤度大的个体
            max_candidate = myfind(max(c_obj_distance[min_candidate]),c_obj_distance[min_candidate])
            if len(max_candidate) != 1:
                max_candidate = max_candidate[1]
            f.append(chromosome[candidate[min_candidate[max_candidate]],:])
        else:
            f.append(chromosome[candidate[min_candidate[0]],:])
    row=len(f)
    F=np.array([[0.0]*variables for i in range(row)])
    for i in range(row):
        for j in range(variables):
            F[i,j]=f[i][j]
    
    return F  #返回数组


def genetic_operator(parent_chromosome, M, V, mu, mum, l_limit, u_limit,graph,saver,sess):
    N=len(parent_chromosome)#N是交配池中的个体数量
    was_crossover = 0#是否交叉标志位
    was_mutation = 0#是否变异标志位
    
    child=[]
    for i in range(N):#这里虽然循环N次，但是每次循环都会有概率产生2个或者1个子代，所以最终产生的子代个体数量大约是2N个
        if (random.random() < 0.9):#交叉概率0.9
            
            parent_1 = round(N*random.random())
            if (parent_1 == N):
                parent_1 = N-1
            
            parent_2 = round(N*random.random())
            if (parent_2 == N):
                parent_2 = N-1
            
            while ((parent_chromosome[parent_1,:]==parent_chromosome[parent_2,:]).all()):
                parent_2 = round(N*random.random())
                if (parent_2 == N):
                    parent_2 = N-1
                
            parent_1 = parent_chromosome[parent_1,:]
            parent_2 = parent_chromosome[parent_2,:]
                     
            u = np.array([0.0 for p in range(V)])
            bq= np.array([0.0 for p in range(V)])
            child_1 = np.array([0.0 for i in range(V+M)])
            child_2 = np.array([0.0 for i in range(V+M)])
            
            for j in range(V):
                u[j] = random.random()
                if (u[j] <= 0.5):
                    bq[j] = math.pow(2*u[j],1/(mu+1))
                else:
                    bq[j] = math.pow(1/(2*(1 - u[j])),1/(mu+1))
               
                child_1[j] = 0.5*(((1 + bq[j])*parent_1[j]) + (1 - bq[j])*parent_2[j])
                child_2[j] = 0.5*(((1 - bq[j])*parent_1[j]) + (1 + bq[j])*parent_2[j])
                if (child_1[j] > u_limit[j]):
                    child_1[j] = u_limit[j]
                elif(child_1[j] < l_limit[j]):
                    child_1[j] = l_limit[j]
                
                if (child_2[j] > u_limit[j]):
                    child_2[j] = u_limit[j]
                elif(child_2[j] < l_limit[j]):
                    child_2[j] = l_limit[j]

            t1 = evaluate_objective(child_1, M, V,graph,saver,sess)
            t2 = evaluate_objective(child_2, M, V,graph,saver,sess)
            for p in range(M):
                child_1[V + p]=t1[p]
                child_2[V + p]=t2[p]
                
            was_crossover = 1
            was_mutation = 0
        else:#if >0.9
            parent_3 = round(N*random.random())
            if (parent_3 ==N):
                parent_3 = N-1
            
            child_3 = parent_chromosome[parent_3]
            
            r = np.array([0.0 for i in range(V)])
            delta= np.array([0.0 for i in range(V)])
            
            for j in range(V):
               r[j] = random.random()
               if (r[j] < 0.5):
                   delta[j] = math.pow(2*r[j],1/(mum+1)) - 1
               else:
                   delta[j] = 1 - math.pow(2*(1 - r[j]),1/(mum+1))
               
               child_3[j] = child_3[j] + delta[j]
               if (child_3[j] > u_limit[j]): # 条件约束
                   child_3[j] = u_limit[j]
               elif (child_3[j] < l_limit[j]):
                   child_3[j] = l_limit[j]
        
            t3 = evaluate_objective(child_3, M, V,graph,saver,sess)
            for p in range(M):
                child_3[V + p]=t3[p]
           
            was_mutation = 1
            was_crossover = 0
        
        if (was_crossover):#交叉
            child.append(child_1) 
            child.append(child_2) 
            was_crossover = 0
            
        elif (was_mutation):#变异
            child.append(child_3) 
            was_mutation = 0
            
    row = len(child)
    f=np.array([[0.0]*(V+M) for i in range(row)])
    for i in range(row):
        for j in range(M+V):
            f[i,j]=child[i][j]
    
    return f      

def replace_chromosome(intermediate_chromosome, M, V,pop):#精英选择策略
    N = len(intermediate_chromosome)
    col=len(intermediate_chromosome[0])
    index=np.argsort(intermediate_chromosome[:,M + V])#倒数第二个
    
    sorted_chromosome=np.array([[0.0]*col for i in range(N)])
    for i in range(N):
        for q in range(col):
            sorted_chromosome[i,q]=intermediate_chromosome[index[i],q]
    
     
    max_rank = int(max(intermediate_chromosome[:,M + V]))
#    print(intermediate_chromosome[:,M + V])
    previous_index = 0
    f=[]
    
    for i in range(max_rank):
        current_index = max(myfind(i+1,sorted_chromosome[:,M + V]))
        if (current_index > pop):
            remaining = pop - previous_index
            temp_pop = sorted_chromosome[previous_index : current_index, :]
            temp_sort_index=np.argsort(-temp_pop[:,M + V + 1])#降序
   
            F= np.array([[0.0]*col for i in range(previous_index+remaining)])
            s_c=0
            
            for u in range(len(f)):
                for v in range(len(f[u])):
                    for w in range(len(f[u][0])):#len(f[u][0])=col
                        F[s_c,w]=f[u][v][w]
                    s_c=s_c+1

            for j in range(remaining):
                for k in range(len(temp_pop[0])):
                    F[s_c+j,k]=temp_pop[temp_sort_index[j],k]
            return F
        elif (current_index < pop):
            f.append(sorted_chromosome[previous_index : current_index, :])
        else:
            f.append(sorted_chromosome[previous_index : current_index, :])
            remaining= current_index - previous_index
            F= np.array([[0.0]*col for i in range(previous_index+remaining)])
            s_c=0;
            for u in range(len(f)):
                for v in range(len(f[u])):
                    for w in range(len(f[u][0])):#len(f[u][0])=col
                        F[s_c,w]=f[u][v][w]
                    s_c=s_c+1   
            return F
        previous_index = current_index
    
    
def evaluate_objective(x, M, V,graph,saver,sess):#计算每个个体的M个目标函数值x(V*1)
     with sess.as_default():
         with graph.as_default():                
            graph_mlp = tf.get_default_graph()
     
            test_x=[]
            test_y=[]
            p=[]
            for i in range(V):
                p.append(x[i])
            test_x.append(p)
            q=[]
            for i in range(M):
                q.append(M_y[i])
            test_y.append(q)
            feed_dict = {"X:0": test_x, "Y:0": test_y}
            prediction = graph_mlp.get_tensor_by_name("prediction:0")
            prediction_value=sess.run(prediction, feed_dict=feed_dict).tolist()
            #print(prediction_value)
            f = np.array([1.0 for i in range(out_num)])
            for i in range(out_num):
                f[i]=1-abs(test_y[0][i]-prediction_value[0][i])
       
            return f
    
    
def nsga_2_optimization(graph,saver,sess):
    pop = 100 #种群数量
    gen = 1 #迭代次数
    M = out_num #目标函数数量
    V = in_num #维度（决策变量的个数）
    min_range = [0 for i in range(V)] #下界 生成1*30的个体向量 全为0
    max_range = [1 for i in range(V)] #上界 生成1*30的个体向量 全为1
       
    chromosome = initialize_variables(pop, M, V, min_range, max_range,graph,saver,sess) #初始化种群(N*(V+M))
    chromosome = non_domination_sort_mod(chromosome, M, V) #对初始化种群进行非支配快速排序和拥挤度计算N*(M+V+2)
    
    for i in range(gen):
        pool = round(pop/2)        #round() 四舍五入取整 交配池大小
        tour = 2                   #竞标赛  参赛选手个数
        parent_chromosome = tournament_selection(chromosome, pool, tour)#竞标赛选择适合繁殖的父代N*(M+V+2)
        mu = 20                    #交叉和变异算法的分布指数
        mum = 20
        offspring_chromosome = genetic_operator(parent_chromosome,M, V, mu, mum, min_range, 
                                                max_range,graph,saver,sess)
                                  #进行交叉变异产生子代 该代码中使用模拟二进制交叉和多项式变异 采用实数编码
        main_pop=len(chromosome)  #父代种群的大小
        offspring_pop=len(offspring_chromosome) #子代种群的大小 
        intermediate_chromosome = np.array([[0.0]*(M+V+2) for i in range(main_pop + offspring_pop)])
        
        for p in range(main_pop):
            for q in range(len(chromosome[0])):
                #print(len(chromosome[0]))
                intermediate_chromosome[p,q]=chromosome[p,q]
        
        for p in range(offspring_pop):
            for q in range(M+V):
                intermediate_chromosome[main_pop+p,q]= offspring_chromosome[p,q]#合并父代种群和子代种群
        
        intermediate_chromosome = non_domination_sort_mod(intermediate_chromosome, M, V)#对新的种群进行快速非支配排序
        chromosome = replace_chromosome(intermediate_chromosome, M, V, pop)#选择合并种群中前N个优先的个体组成新种群
        
#        if ((i+1)%10==0):
#            print("%d generations completed\n"%(i+1))
        
    #Mx=chromosome[:,V ]
    #My=chromosome[:,V + 1]
    return chromosome

def MainIn(graph,saver,sess):
        result=nsga_2_optimization(graph,saver,sess)
        print("Finished！")
        #写入文件
    
        global MaxV
        f = np.array([1.0 for i in range(in_num)])
        for i in range(in_num):
            f[i]=result[0][i]*MaxV[i]
        return f


            
    
    
    
    
    