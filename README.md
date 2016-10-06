# OpFlowCNN
Optical flow estimation using convolutional neural network

## Run training
Dataset for training can be found in data/synt-mat
Custom dataset can be created using matlab script dataset_m.m

### Arguments for training
* CUDA_VISIBLE_DEVICES - selecting gpu to run training on
* data_dir - where is placed the dataset to be used for training and validation
* result_dir - directory for the results of the training
* sgd - decides if training will use sgd or adadelta (1 = sgd)
* model - select from models (1 = model-small, 2 = model-basic, 3 = model-parallel) 
* batchSize - size of the batch
* print_freq - how often will the results be saved
* print_img - should the results be saved
* save_model - should the model be saved (will be saved in models-learned)
* dataSize - size of the training set
* valSize - size of the validation set

### Train using OpticalFlowCriterion
CUDA_VISIBLE_DEVICES=0 th train-gpu.lua -data_dir data/ -result_dir 10_06 -sgd 1 -model 2 -batchSize 10 -epochs 3000 -print_freq 100 -print_img 1 -save_model 0 -dataSize 900 -valSize 100

### Train using MSECriterion (Mean square error)
CUDA_VISIBLE_DEVICES=0 th train-mse.lua -data_dir data/ -result_dir 10_06-mse -sgd 1 -model 2 -batchSize 1 -epochs 5000 -print_freq 100 -print_img 1 -save_model 1 -dataSize 900 -valSize 100

