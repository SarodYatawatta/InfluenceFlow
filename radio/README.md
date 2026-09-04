## Requirements
Hardware:
A GPU is essential.

The following executables are required:

- Python libraries: PyTorch, numpy, scipy, astropy
- For creation of blank MS [makems](https://git.astron.nl/ro/lofar/-/blob/master/CEP/MS/src/makems.cc)
- For generating data, calibration, influence function calculation (use CUDA build) [sagecal](https://github.com/nlesc-dirac/sagecal)

After building the above software, edit ```./generate_data.py``` to point to their correct locations in your installation.

## Running
Common settings:

```
HIDDEN=5500
DEPTH=1
ITER=150000
WARMUP=10000
BATCHSIZE=256
BATCHNRM=0
ERRBOUND=0.90
CONSTRAINT=0
REG=1
```

Generate data and perform pre-training:

```
./train_eval.py --load_data 1 --data_size 50000 --batch_size $BATCHSIZE --regularization 0.1 --DNN_hidden $HIDDEN --iterations $ITER --warmup $WARMUP --evaluations 256 --DNN_depth $DEPTH --batchnorm 0 --error_bound $ERRBOUND  --load_model 0 --initial_training 1
```

Final training:

```
./train_eval.py --load_data 1 --data_size 50000 --batch_size $BATCHSIZE --regularization $REG --DNN_hidden $HIDDEN --iterations $ITER --warmup $WARMUP --evaluations 256 --DNN_depth $DEPTH --batchnorm 0 --error_bound $ERRBOUND --load_model 1 --initial_training 0 --constraint_type $CONSTRAINT
```

Evaluate the model with one long observation:

```
NSOL=7200
./eval_model_obs.py --num_sol $NSOL --num_sol_interval 1 --DNN_hidden $HIDDEN --DNN_depth $DEPTH --threshold 100 --frequency 115e6 --seed 45446
```
