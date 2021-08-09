[toc]

## Model parameter description
Unspecified parameters are default parameters

### Xgboost：

learning_rate: 0.1

n_estimators: 500

max_depth: 5

min_child_weight: 1

subsample: 0.8

colsample_bytree: 0.8

 reg_lambda: 1

### GBDT:  

n_estimators: 600

max_depth: 3

min_samples_split: 2

learning_rate: 0.08

### PLSR:  

n_components=12

max_iter=600

### RandomForestRegressor：  

n_estimators=70

### RNN:

rnn_unit = 5

input_size = 32

output_size = 1

lr = 0.0001

### Adaboost:  

DecisionTreeRegressor()

n_estimators=200

learning_rate=0.08

