import math
import numpy
import torch
import sys
def mst(input,pi,point_number):
    list_input = input.cpu().numpy().tolist()
    result = []
    test_result = []
    select_node_result = []
    origin_number = point_number
    better = 0
    for select_node in range(0,origin_number-1):
        globals()["select_list" + str(select_node)] = []
    for data_id in range(len(list_input)):
        current_node=[]
        select_node_number = 0
        for node_id in range(origin_number):
            current_node.append(list_input[data_id][node_id])
        cost = numpy.zeros([len(current_node), len(current_node)])
        for i in range(len(current_node)):
            for j in range(i,len(current_node)):
                temp = (current_node[j][0] -current_node[i][0])*(current_node[j][0] -current_node[i][0]) + (current_node[j][1] -current_node[i][1]) * (current_node[j][1] -current_node[i][1])
                cost[i][j] = math.sqrt(temp)
                cost[j][i] = cost[i][j]
        sum = 0
        mst = Prim(cost, 0,  sum)
        result_storage = []
        min = mst
        globals()["select" + str(0)] = mst
        globals()["select_list" + str(0)].append(globals()["select" + str(0)])
        for select_node in range(0,len(list_input[0])-origin_number):
            current_node.append(list_input[data_id][select_node + origin_number])
            cost = numpy.zeros([len(current_node), len(current_node)])
            for i in range(len(current_node)):
                for j in range(i,len(current_node)):
                    temp = (current_node[j][0] -current_node[i][0])*(current_node[j][0] -current_node[i][0]) + (current_node[j][1] -current_node[i][1]) * (current_node[j][1] -current_node[i][1])
                    cost[i][j] = math.sqrt(temp)
                    cost[j][i] = cost[i][j]
            sum = 0
            #temp = 0
            temp = Prim(cost, 0,  sum)
            result_storage.append(temp)
            globals()["select" + str(select_node+1)] = temp
            globals()["select_list" + str(select_node+1)].append(globals()["select" + str(select_node+1)])
        for select_node in range(len(list_input[0])-origin_number,origin_number-2):
            globals()["select" + str(select_node+1)] = 0
            globals()["select_list" + str(select_node+1)].append(globals()["select" + str(select_node+1)])        
        for i in range(len(result_storage)):
            if(result_storage[i] < min):
                min = result_storage[i]
            else:
                temp_res = result_storage[i]
                test_temp = min
                select_node_number = i + 1
                break
            select_node_number = len(result_storage)
            temp_res = globals()["select" + str(select_node_number)]
            test_temp = min
        if(min < mst):
            better = better + 1
        result.append(temp_res)
        test_result.append(test_temp)
        select_node_result.append(select_node_number)
        origin_a = current_node[0: 10]
        selecte_a = current_node[0: 10+select_node_number]
        origin_afile = open("origin_a.txt", "a")
        origin_afile.write(str(origin_a) + '\n')
        origin_afile.close()
        selecte_afile = open("selecte_a.txt", "a")
        selecte_afile.write(str(selecte_a) + '\n')
        selecte_afile.close()
    result_tensor=torch.cuda.FloatTensor(result)
    test_cost_tensor = torch.cuda.FloatTensor(test_result)
    total_result_vector = []
    for select_node in range(0, point_number - 1):
        total_result_vector.append(globals()["select_list" + str(select_node)])
    return result_tensor, select_node_result, total_result_vector, test_cost_tensor, better

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
