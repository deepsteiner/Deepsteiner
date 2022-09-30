# Deep-Steiner: Learning to Solve the Euclidean Steiner Tree Problem.

A deep reinforcment learning model to solve the Euclidean Steiner Tree (EST) problem. Training with REINFORCE with greedy rollout baseline.

## paper
Implementation of our paper: [Deep-Steiner: Learning to Solve the Euclidean Steiner Tree Problem](https://arxiv.org/abs/2209.09983), which is accepted by [EAI WiCON 2022](https://wicon.eai-conferences.org/2022/)

## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)



## Usage

### Generating data

Training data is generated on the fly. To generate validation and test data for all problems with seed number:
```bash
python generate_data.py --problem all --name validation --seed 4321
python generate_data.py --problem all --name test --seed 1234
```

### Training

For training EST problem instances with 10 nodes and using rollout as REINFORCE baseline and using the generated validation set:
```bash
python run.py --graph_size 10 --batch_size 32 --epoch_size 10240 --val_size 10000 --eval_batch_size 10 --baseline rollout --run_name 'est10' --n_epochs 100 --lr_model 0.00000001 --seed 1111 --embedding_dim 128 --hidden_dim 128 --n_encode_layers 5
```

#### Warm start
You can initialize a run using a pretrained model by using the `--load_path` option:
```bash
python run.py --graph_size 10 --batch_size 32 --epoch_size 10240 --val_size 10000 --eval_batch_size 10 --baseline rollout --run_name 'est10' --n_epochs 100 --lr_model 0.00000001 --seed 1111 --embedding_dim 128 --hidden_dim 128 --n_encode_layers 5 --load_path /content/drive/MyDrive/attention_completeV3.0.1/outputs/tsp_10/arc9/epoch-80.pt
```

### Evaluation
To evaluate a model, you can use the eval.py to output the results. All the generated Steiner tree will be saved in the file "select_a.txt":
```bash
python eval.py --graph_size 10 --batch_size 32 --epoch_size 10240 --val_size 10000 --eval_batch_size 10 --baseline rollout --run_name 'est10' --n_epochs 100 --lr_model 0.00000001 --seed 1111 --embedding_dim 128 --hidden_dim 128 --n_encode_layers 5 --load_path /content/drive/MyDrive/attention_completeV3.0.1/outputs/tsp_10/arc9/epoch-80.pt
```

### Other options and help
```bash
python run.py -h
python eval.py -h
```


## Acknowledgements
Thanks to [ wouterkool / attention-learn-to-route ](https://github.com/wouterkool/attention-learn-to-route#attention-learn-to-solve-routing-problems) for getting me started with the code for the graph attention model.
