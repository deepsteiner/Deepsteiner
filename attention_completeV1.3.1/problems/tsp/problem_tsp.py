from torch.utils.data import Dataset
import torch
import os
import sys
import pickle
import math
import heapq
import random
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
from problems.tsp.mst import mst
from problems.tsp.angle import st_point_generate
from problems.tsp.mst_routine import mst_routine
from math import pi
class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi, point_number):
        result_tensor, select_node_result, total_result_vector, test_tensor, better = mst(dataset,pi,point_number)
        return result_tensor, select_node_result, None, total_result_vector, test_tensor, better

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)


    @staticmethod
    def expend_dataset(input):
        list_dataset = input.cpu().numpy().tolist()
        new_dataset = []
        for data_id in range(len(list_dataset)):
            new_data = []
            data = list_dataset[data_id]
            u = mst_routine(data)
            for i in range(1, len(u)):
                point1 = [0,0]
                point2 = [0,0]
                point1 = data[i]
                point2 = data[int(u[i])]
                dense = 9
                for deg_id in range(dense):
                    deg = pi * (1 / 3 / (dense + 1)) * (deg_id + 1)
                    spoint1 = [0, 0]
                    spoint2 = [0, 0]
                    spoint1[0], spoint1[1], spoint2[0], spoint2[1] = st_point_generate(point1[0], point1[1], point2[0], point2[1],deg)
                    new_data.append(spoint1)
                    new_data.append(spoint2)
            new_dataset.append(new_data)
        dataset_tensor=torch.cuda.FloatTensor(new_dataset)
        return dataset_tensor.to('cuda:0')

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]            
            #self.data = [torch.normal(0.5,0.5,size=(10,2)) for i in range(num_samples)]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
