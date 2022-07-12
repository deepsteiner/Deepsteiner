import os
import time
from tqdm import tqdm
import torch
import math
import heapq
import random

from problems.tsp.angle import st_point_generate
from problems.tsp.mst_instance import mst_instance
from problems.tsp.mst_routine import mst_routine
from math import pi
def random_guess(bat,select_number_vector):
        point_number = len(bat[0])
        origin_dataset = bat
        list_original = origin_dataset.cpu().numpy().tolist()
        random_guess_return = []
        for i in range(len(bat)):
            temp_mst = mst_instance(list_original[i])
            original_graph = list_original[i]
            for j in range(select_number_vector[i]-1):
                candidate = expend_graph(original_graph)
                random_choice = random.randint(0,len(candidate)-1)
                original_graph.append(candidate[random_choice])

            random_guess_result = mst_instance(original_graph)
            random_guess_return.append(random_guess_result)

        random_result_tensor=torch.cuda.FloatTensor(random_guess_return)
        return  random_result_tensor

def expend_graph(original_graph):
    new_data = []
    u = mst_routine(original_graph)
    for i in range(1, len(u)):
        point1 = [0,0]
        point2 = [0,0]
        point1 = original_graph[i]
        point2 = original_graph[int(u[i])]
        dense = 9
        for deg_id in range(dense):
            deg = pi * (1 / 3 / (dense + 1)) * (deg_id + 1)
            spoint1 = [0, 0]
            spoint2 = [0, 0]
            spoint1[0], spoint1[1], spoint2[0], spoint2[1] = st_point_generate(point1[0], point1[1], point2[0], point2[1],deg)
            new_data.append(spoint1)
            new_data.append(spoint2)
    return new_data
