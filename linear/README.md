Generate data and train the model:
```
./train_eval_xcorr.py --model_N 15 --model_M 10 --data_size 40000 --iterations 150000 --load_data 0 --DNN_hidden 256 --DNN_depth 5 --regularization 50 --initial_training 0 --bound 1.1 --seed 99
```

Evaluate the trained model:
```
./eval_model.py --model_N 15 --model_M 10 --DNN_hidden 256 --DNN_depth 5 --seed 99 --evaluations 3000
```
