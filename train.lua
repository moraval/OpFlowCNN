-- training the model
require('cutorch')
require('nn')
require('cunn')
--require('cudnn')
require('optim')
require('paths')
require('nngraph')

require 'optim'
--require 'dataset'
require 'OpticalFlowCriterion'
require 'model/model.lua'
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

local batchInputs = torch.load(data_dir .. 'train_data.t7', 'ascii')
local batchLabels = torch.load(data_dir .. 'train_labels.t7', 'ascii')

--local batchInputs = {}
--local batchLabels = {}

--size1 = batchInputs[1]:size()
--print(size1)
--print(batchInputs:size())
--print(batchLabels:size())

----------------------------------------------------------------------

local criterion = nn.OpticalFlowCriterion
local params, gradParams = model:getParameters() -- to flatten all the matrices inside the model
local optimState = {learningRate=0.1}

for epoch=1,5 do
  local function feval(params)
    gradParams:zero()
    
    print("starting")
    
    local outputs = model:forward(batchInputs)
    
    print("forward done")
    
    local loss = criterion:forward(outputs, batchInputs)
    local dloss_doutput = criterion:backward(outputs, batchInputs)
    
    print("loss " .. loss)
    print('example of gradient: ')
    print(dloss_doutput[1][1][75][200]..', '..dloss_doutput[1][1][76][200]..', '..dloss_doutput[1][1][77][200])
    print(dloss_doutput[2][1][100][520]..', '..dloss_doutput[2][1][100][521]..', '..dloss_doutput[2][1][1][522])
    print(dloss_doutput[1][2][150][800]..', '..dloss_doutput[1][2][151][801]..', '..dloss_doutput[1][2][152][802])
    
    model:backward(batchInputs, dloss_doutput)
    return loss,gradParams
  end
  optim.sgd(feval, params, optimState)
end

