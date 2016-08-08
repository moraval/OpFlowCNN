-- training the model

require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

-- configurations
cutorch.setDevice(1)
--torch.setnumthreads(4)
--torch.setdefaulttensortype('torch.FloatTensor')
local model_dir = "model"
setupLogger(paths.concat(model_dir, 'log.txt'))
paths.dofile(paths.concat(model_dir, 'model.lua'))
gConfig = getConfig()
gConfig.model_dir = model_dir

-- `createModel` is defined in config.lua, it returns the network model and the criterion (loss function)
local model, criterion = createModel(gConfig)
logging(string.format('Model configuration:\n%s', model))
local modelSize, nParamsEachLayer = modelSize(model)
logging(string.format('Model size: %d\n%s', modelSize, nParamsEachLayer))


require 'optim'

for epoch=1,50 do
  -- local function we give to optim
  -- it takes current weights as input, and outputs the loss
  -- and the gradient of the loss with respect to the weights
  -- gradParams is calculated implicitly by calling 'backward',
  -- because the model's weight and bias gradient tensors
  -- are simply views onto gradParams
  local function feval(params)
    gradParams:zero()

    local outputs = model:forward(batchInputs)
    local loss = criterion:forward(outputs, batchLabels, batchInputs)
    local dloss_doutput = criterion:backward(outputs, batchLabels, batchInputs)
    model:backward(batchInputs, dloss_doutput)

    return loss,gradParams
  end
  optim.sgd(feval, params, optimState)
end