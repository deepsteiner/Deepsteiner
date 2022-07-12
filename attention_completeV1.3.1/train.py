import os
import time
from tqdm import tqdm
import torch
import math
import heapq
import random

from torch.utils.data import DataLoader
from torch.nn import DataParallel
from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
from problems.tsp.mst_instance import mst_instance
from utils.random_guess import random_guess

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def eval_model_bat_baseline(bat, baseline, cost):
    bl_val, bl_loss = baseline.eval(bat, cost)
    return bl_val.data.cpu()

def test(model, dataset, baseline, opts):
    print('Test...')
    cost, random_guess = rollout_test(model, dataset, opts)
    bl_sum = 0
    for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=1):
      bl_val = eval_model_bat_baseline(bat, baseline, cost)
      bl_sum = bl_sum + bl_val
    avg_cost = cost.mean().item()
    avg_random = random_guess.mean().item()
    avg_baseline = bl_sum.mean()/1000
    testfile = open("testfile.txt", "a")
    testfile.write(str(avg_cost) + '\n')
    testfile.close()
    print('Test overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    randomfile = open("randomfile.txt", "a")
    randomfile.write(str(avg_random) + '\n')
    randomfile.close()
    print('random overall avg_cost: {} +- {}'.format(
        avg_random, torch.std(random_guess) / math.sqrt(len(random_guess))))
    sys.exit()
    testrefile = open("testrefile.txt", "a")
    testrefile.write(str((avg_cost - avg_baseline)*1000) + '\n')
    testrefile.close()
    print('reinforce_loss overall: {}'.format(
        (avg_cost - avg_baseline)*1000))

    return avg_cost

def validate(model, dataset, baseline, opts):
    # Validate
    print('Validate...')
    cost, random_guess = rollout(model, dataset, opts)
    bl_sum = 0
    for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=1):
      bl_val = eval_model_bat_baseline(bat, baseline, cost)
      bl_sum = bl_sum + bl_val
    avg_cost = cost.mean().item()
    avg_random = random_guess.mean().item()
    avg_baseline = bl_sum.mean()/1000
    testfile = open("Validatefile.txt", "a")
    testfile.write(str(avg_cost) + '\n')
    testfile.close()
    print('Validate overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    randomfile = open("randomValidatefile.txt", "a")
    randomfile.write(str(avg_random) + '\n')
    randomfile.close()
    print('random overall avg_cost: {} +- {}'.format(
        avg_random, torch.std(random_guess) / math.sqrt(len(random_guess))))
    testrefile = open("Validaterefile.txt", "a")
    testrefile.write(str((avg_cost - avg_baseline)*1000) + '\n')
    testrefile.close()
    print('reinforce_loss overall: {}'.format(
        (avg_cost - avg_baseline)*1000))

    return avg_cost

def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    tag = 0
    finalbetter = 0
    def eval_model_bat_cost(bat):
        with torch.no_grad():
            _, _, select_node_vector, total_result_vector, cost, better = model(move_to(bat, opts.device))
        random_result = random_guess(bat, select_node_vector)
        #print(total_result_tensor)
        total_result_tensor=torch.cuda.FloatTensor(total_result_vector)
        return cost.data.cpu(),random_result.data.cpu(),total_result_tensor.data.cpu(), better,select_node_vector
    """
    def eval_model_bat_random(bat):
        with torch.no_grad():
            cost, _, select_node_vector, total_result_vector = model(move_to(bat, opts.device))
        random_result = random_guess(bat, select_node_vector)
        return random_result.data.cpu()
    """
    for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=1):
      evadata = open("evadata.txt", "a")
      evadata.write(str(bat))
      evadata.close()
      cost, random_guess_res, total_result_tensor,better,select_node_vector = eval_model_bat_cost(bat)
      for i in range(len(cost)):
            valcost = open("realcost.txt", "a")
            valcost.write(str(cost[i]) + '\n')
            valcost.close()
      finalbetter = finalbetter + better
      if tag == 0:
        final_cost = cost
        final_random = random_guess_res
        final_total = total_result_tensor
        final_select_node_vector = select_node_vector
        tag = 1
      else:
        final_cost = torch.cat((final_cost,cost),0)
        final_random = torch.cat((final_random,random_guess_res),0)
        final_total = torch.cat((final_total,total_result_tensor),1)
        final_select_node_vector = final_select_node_vector + select_node_vector
    valibetter = open("valibetter.txt", "a")
    valibetter.write(str(finalbetter) + '\n')
    valibetter.close()
    valiselect = open("valiselect.txt", "a")
    valiselect.write(str(final_select_node_vector) + '\n')
    valiselect.close()
    """
    cost, random_guess_res = torch.cat([
        eval_model_bat_cost(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=1)
    ], 0)
    sys.exit()
    random_guess_res = torch.cat([
        eval_model_bat_random(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=1)
    ], 0)
    """
    return final_cost,final_random

def rollout_test(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    tag = 0
    finalbetter = 0
    def eval_model_bat_cost(bat):
        testdata = open("datset.txt", "a")
        testdata.write(str(bat))
        testdata.close()
        with torch.no_grad():
          times = time.time()
          _, _, select_node_vector, total_result_vector,cost ,better = model(move_to(bat, opts.device))
          timet = time.time()
          testtime = open("testtime.txt", "a")
          testtime.write(str(timet-times) + '\n')
          testtime.close()
          for i in range(len(cost)):
            testcost = open("testcost.txt", "a")
            testcost.write(str(cost[i]) + '\n')
            testcost.close()
            testnode = open("testnode.txt", "a")
            testnode.write(str(select_node_vector[i]) + '\n')
            testnode.close()
        rtimes=time.time()
        random_result = random_guess(bat, select_node_vector)
        rtimee=time.time()
        rtesttime = open("randomtesttime.txt", "a")
        rtesttime.write(str(rtimee-rtimes) + '\n')
        rtesttime.close()
        for i in range(len(random_result)):
          randomtestcost = open("randomtestcost.txt", "a")
          randomtestcost.write(str(random_result[i]) + '\n')
          randomtestcost.close()
        total_result_tensor=torch.cuda.FloatTensor(total_result_vector)
        return cost.data.cpu(),random_result.data.cpu(),total_result_tensor.data.cpu(), better, select_node_vector

    for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=1):
      cost, random_guess_res, total_result_tensor,better,select_node_vector = eval_model_bat_cost(bat)
      finalbetter = finalbetter + better
      if tag == 0:
        final_cost = cost
        final_random = random_guess_res
        final_total = total_result_tensor
        final_select_node_vector = select_node_vector
        tag = 1
      else:
        final_cost = torch.cat((final_cost,cost),0)
        final_random = torch.cat((final_random,random_guess_res),0)
        final_total = torch.cat((final_total,total_result_tensor),1)
        final_select_node_vector = final_select_node_vector + select_node_vector
    # testbetter = open("testbetter.txt", "a")
    # testbetter.write(str(finalbetter) + '\n')
    # testbetter.close()
    # testselect = open("testselect.txt", "a")
    # testselect.write(str(final_select_node_vector) + '\n')
    # testselect.close()
    # for i in range(0,9):
    #   try:
    #       final_total[i]
    #   except IndexError:
    #       trainrefile_select = open("test_select%s"%(i) + ".txt", "a")
    #       trainrefile_select.write(str(0) + '\n')
    #       trainrefile_select.close()
    #   else:
    #       trainrefile_select = open("test_select%s"%(i) + ".txt", "a")
    #       a = torch.count_nonzero(final_total[i]).item()
    #       b = torch.sum(final_total[i]).item()
    #       avg = 0
    #       if(a != 0):
    #         avg = b/a
    #       trainrefile_select.write(str(avg) + '\n')
    #       trainrefile_select.close()
    """
    cost, random_guess_res = torch.cat([
        eval_model_bat_cost(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=1)
    ], 0)
    sys.exit()
    random_guess_res = torch.cat([
        eval_model_bat_random(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=1)
    ], 0)
    """
    return final_cost,final_random


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, only_dataset, test_dataset):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(only_dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    loss_sum = 0
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        loss_single = train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            val_dataset,
            opts
        )
        #print("The batch Loss is {},".format(loss_single))
        loss_sum = loss_sum + loss_single
        step += 1
        if(batch_id % (1280//opts.batch_size) == 0):
          test(model, test_dataset, baseline, opts)
        model.train()
        set_decode_type(model, "sampling")
    print("The total Loss is {},".format(loss_sum))
    epoch_duration = time.time() - start_time
    print("Loss is {},".format(loss_sum))
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, baseline,opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        val_dataset,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    # Evaluate model, get costs and log probabilities
    cost, log_likelihood, select_node_vector,total_result_vector,_, better = model(x)
    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss
    # print("Reinforce Loss is {},".format(loss))
    # calculate the result for selecting 1 to n-3 points
    # total_result_vector size is [n-3,batch_size], we now use 7
    # if step % (1280//opts.batch_size) == 0:
    #   train_select_point = open("train_select_point.txt", "a")
    #   train_select_point.write(str(select_node_vector))
    #   train_select_point.close()
    #   trainrefile = open("trainrefile.txt", "a")
    #   trainrefile.write(str(loss) + '\n')
    #   trainrefile.close()
    #   trainbetter = open("trainbetter.txt", "a")
    #   trainbetter.write(str(better) + '\n')
    #   trainbetter.close()
    #   for i in range(9):
    #     try:
    #       total_result_vector[i]
    #     except IndexError:
    #       trainrefile_select = open("train_select%s"%(i) + ".txt", "a")
    #       trainrefile_select.write(str(0) + '\n')
    #       trainrefile_select.close()
    #     else:
    #       trainrefile_select = open("train_select%s"%(i) + ".txt", "a")
    #       avg = sum(total_result_vector[i])/len(total_result_vector[i])
    #       trainrefile_select.write(str(avg) + '\n')
    #       trainrefile_select.close()
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()


    # Logging
    if step % (1280//opts.batch_size) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
    return loss
