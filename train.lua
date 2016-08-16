-- training the model
require('cutorch')
require('nn')
require('cunn')
--require('cudnn')
require('optim')
require('paths')
require('nngraph')
require 'nn'

require 'optim'
--require 'dataset'
require 'nn/OpticalFlowCriterion'
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
local batchSize = arg[2]

-- WHICH ONE is CORRECT?
local allData = torch.load(data_dir .. 'train_data_small.t7', 'ascii')
--local batchLabels = torch.load(data_dir .. 'train_labels_small.t7', 'ascii')

----------------------------------------------------------------------

local criterion = nn.OpticalFlowCriterion
--local criterion = nn.ClassNLLCriterion
local params, gradParams = model:getParameters() -- to flatten all the matrices inside the model
local epochs = arg[3]
local optimState = {}
nrOfBatches = 4


for epoch=1,epochs do
  for batch = 0,nrOfBatches-1 do
    local start = batch * batchSize + 1
    local myend = start + batchSize -1
    local batchInputs = allData:sub(start, myend)
    
    local function feval(params)
      gradParams:zero()

      print("\nSTARTING EPOCH "..epoch)

      local outputs = model:forward(batchInputs)

      print("\nforward done")

      local loss = criterion:forward(outputs, batchInputs)
      local dloss_doutput = criterion:backward(outputs, batchInputs)

      print("\nLOSS " .. loss)

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
--  optim.adadelta(feval, params)
  end
end
