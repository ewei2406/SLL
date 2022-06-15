# SLL
Selective Learnability Lock

Prequisites: install `requirements.txt`

To execute Selective Learnability Lock on a dataset, navigate to Experiment/SelectiveLL and run the script:

```
python3 SelectiveLL.py --datatset DATASET --ptb_rate 0.25
```

where DATASET can be `cora, citeseer, Polblogs, flickr, Blogcatalog`. See all arguments with `python3 SelectiveLL.py -h`:

```
usage: SelectiveLL.py [-h] [--dataset DATASET] [--seed SEED]
                      [--model_lr MODEL_LR] [--weight_decay WEIGHT_DECAY]
                      [--hidden_layers HIDDEN_LAYERS] [--dropout DROPOUT]
                      [--protect_size PROTECT_SIZE] [--ptb_rate PTB_RATE]
                      [--do_sampling DO_SAMPLING] [--sample_size SAMPLE_SIZE]
                      [--num_samples NUM_SAMPLES] [--ptb_epochs PTB_EPOCHS]
                      [--save SAVE] [--save_location SAVE_LOCATION]
                      [--save_perturbations SAVE_PERTURBATIONS]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset
  --seed SEED           Random seed for model
  --model_lr MODEL_LR   Initial learning rate
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 loss on parameters)
  --hidden_layers HIDDEN_LAYERS
                        Number of hidden layers
  --dropout DROPOUT     Dropout rate for GCN
  --protect_size PROTECT_SIZE
                        Number of randomly chosen protected nodes
  --ptb_rate PTB_RATE   Perturbation rate (percentage of available edges)
  --do_sampling DO_SAMPLING
                        To do sampling or not during SLL
  --sample_size SAMPLE_SIZE
                        The size of each sample
  --num_samples NUM_SAMPLES
                        The number of samples
  --ptb_epochs PTB_EPOCHS
                        Number of epochs to perform SLL
  --save SAVE           Save the outputs to csv
  --save_location SAVE_LOCATION
                        Where to save the outputs to csv
  --save_perturbations SAVE_PERTURBATIONS
                        Save the perturbation matrix to temp var
```