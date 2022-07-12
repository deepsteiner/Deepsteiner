import math
import numpy

def mst_instance(current_node):
        cost = numpy.zeros([len(current_node), len(current_node)])
        for i in range(len(current_node)):
            for j in range(i,len(current_node)):
                temp = (current_node[j][0] -current_node[i][0])*(current_node[j][0] -current_node[i][0]) + (current_node[j][1] -current_node[i][1]) * (current_node[j][1] -current_node[i][1])
                cost[i][j] = math.sqrt(temp)
                cost[j][i] = cost[i][j]
        sum = 0
        result = Prim(cost, 0,  sum)
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
