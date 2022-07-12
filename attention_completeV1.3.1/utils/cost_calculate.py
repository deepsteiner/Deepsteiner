import math
import numpy
import torch

def cost_compare(old, new):
    flag = 0
    for i in range(len(old)):
        if(old[i] > new[i]):
            old[i] = new[i]
            flag = 1
    return old, flag

def cost_calculate(input):
    list_input = input.cpu().numpy().tolist()
    result = []
    for data_id in range(len(list_input)):
        current_data = list_input[data_id]
        cost = numpy.zeros([len(current_data), len(current_data)])
        for i in range(len(current_data)):
            for j in range(i,len(current_data)):
                temp = (current_data[j][0] -current_data[i][0])*(current_data[j][0] -current_data[i][0]) + (current_data[j][1] -current_data[i][1]) * (current_data[j][1] -current_data[i][1])
                cost[i][j] = math.sqrt(temp)
                cost[j][i] = cost[i][j]
        sum = 0
        mst = Prim(cost, 0,  sum)
        result.append(mst)
    return result

def Prim (V, vertex, sum):
    length = len(V);
    lowcost = numpy.zeros([length])
    U = numpy.zeros([length])

    for i in range(length):
        lowcost[i] = V[vertex, i]
        U[i] = vertex
        lowcost[vertex] = -1;
    for i in range(1,length):
            k = 0
            min = 65535
            for j in range(length):
                if((lowcost[j] > 0) & (lowcost[j] < min)):
                    min = lowcost[j]
                    k = j
            lowcost[k] = -1
            sum = sum + min
            for j in range(length):
                if((V[k, j] != 0) & ((lowcost[j] == 0 )| (V[k, j] < lowcost[j]))):
                    lowcost[j] = V[k, j]
                    U[j] = k
    return sum
