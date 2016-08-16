-- training the model
require('cutorch')
require('nn')
--require('cunn')
--require('cudnn')
require('optim')
require('paths')
require('nngraph')
require 'nn'

require 'optim'
--require 'dataset'
--require 'nn/OpticalFlowCriterion'
require 'OpticalFlowCriterion'
require './model/model.lua'
----------------------------------------------------------------------
--local model_dir = "model"

--setupLogger(paths.concat(model_dir, 'log.txt'))
--paths.dofile(paths.concat(model_dir, 'model.lua'))

-- `createModel` is defined in model.lua, it returns the network model and the criterion (loss function)
local model = create_model()

--logging(string.format('Model:\n%s', model))
print("Model:")
print(model)

----------------------------------------------------------------------
-- dataset 
local data_dir = arg[1]

--local batchSize = 128
local batchSize = arg[2]
--local batchInputs, batchLabels = create_dataset()

-- WHICH ONE is CORRECT?
--local trainSet = torch.load(data_dir .. 'train_data.t7', 'ascii')

local batchInputs = torch.load(data_dir .. 'train_data_small.t7', 'ascii')/255
--local batchLabels = torch.load(data_dir .. 'train_labels_small.t7', 'ascii')

--local batchInputs = {}
--local batchLabels = {}

--size1 = batchInputs[1]:size()
--print(size1)
--print(batchInputs:size())
--print(batchLabels:size())

----------------------------------------------------------------------

local criterion = nn.OpticalFlowCriterion
--local criterion = nn.ClassNLLCriterion
local params, gradParams = model:getParameters() -- to flatten all the matrices inside the model
local epochs = arg[3]
local optimState = {}

for epoch=1,epochs do
  local function feval(params)
    gradParams:zero()

    print("\nSTARTING EPOCH "..epoch)

    local outputs = model:forward(batchInputs)

    print("\nforward done")

    local loss = criterion:forward(outputs, batchInputs)
    
    print("\nLOSS " .. loss)
    
    local dloss_doutput = criterion:backward(outputs, batchInputs)

    my_string = ''
    for r=1,94,30 do
      for s=1,311,60 do
        my_string = my_string..dloss_doutput[1][1][r][s]..' '
      end
    end
    print('grads 1')
    print(my_string)

    model:backward(batchInputs, dloss_doutput)
    return loss,gradParams
  end
  optim.adadelta(feval, params, optimState)
end

