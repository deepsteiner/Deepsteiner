import numpy
import torch
import random
def random_select(input):
    list_input = input.cpu().numpy().tolist()
    result = []
    for i in range(len(list_input)):
        max_number = max(list_input[i])
        possible_select = []
        for j in range(len(list_input[i])):
            if(list_input[i][j] == max_number):
                possible_select.append(j)
        select = random.randint(1,len(possible_select)) - 1
        result.append(possible_select[select])
    result_tensor=torch.cuda.LongTensor(result)            
    return result_tensor