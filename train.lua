-- training the model
require('cutorch')
require('nn')
require('cunn')
--require('cudnn') -- libcudnn missing
require('optim')
require('paths')
require('nngraph')
require('sys')

require 'optim'
--require 'nn/OpticalFlowCriterion'
require 'OpticalFlowCriterion'
require 'model/model.lua'
require 'os'
require 'image'
require 'synth_dataset'

local data_dir = arg[1]
local trainingData = arg[2]
local targetData = arg[3]
local resDir = arg[4]
local batchSize = arg[5]
local epochs = arg[6]
local printF = arg[7]
local coarse2F = arg[8]
local normalize = arg[9]
local channels = 3
local BS = 1
local S1 = 16
local S2 = 16

----------------------------------------------------------------------
-- dataset 

--local trainData = torch.load(data_dir .. trainingData, 'ascii')/normalize
--local targData = torch.load(data_dir .. targetData, 'ascii'):sub(1,trainData:size(1))/normalize

--local trainData, targData, GT = create_dataset(resDir)
local trainData, targData, GT = load_dataset()

print(targData:size())
print(trainData:size())

local nrOfBatches = trainData:size(1)/batchSize
local size1 = trainData:size(3)
local size2 = trainData:size(4)

for i = 1, batchSize * nrOfBatches do
  local j = math.random(i, batchSize*nrOfBatches)
  trainData[i], trainData[j] = trainData[j], trainData[i]
  targData[i], targData[j] = targData[j], targData[i]
end

----------------------------------------------------------------------
-- `create_model` is defined in model.lua, it returns the network model

local model = create_model(channels, size1, size2)
print("Model:")
print(model)

----------------------------------------------------------------------
-- Trying to optimize memory

opts = {inplace=true, mode='training'}
optnet = require 'optnet'
optnet.optimizeMemory(model, trainData:sub(1, batchSize):cuda(), opts)
print('done optimization')
----------------------------------------------------------------------

local losses_name = 'results/'..resDir .. '/losses/losses_all.csv'
local out = assert(io.open(losses_name, "w")) -- open a file for serialization

for a = 0.5,0.5,0.3 do
  local losses = torch.Tensor(4)
  local print_A = a*10

  out:write(print_A)
  out:write(',')

  a = 0.03
  local criterion = nn.OpticalFlowCriterion(resDir, channels, batchSize, printF, a, normalize, GT)
  local params, gradParams = model:getParameters() -- to flatten all the matrices inside the model

  local optimState = {}
  config = {
    learningRate = 1e-3,
    momentum = 0.5
  }
----------------------------------------------------------------------

  for epoch=1,epochs do

    print("STARTING EPOCH "..epoch)

    for batch = 0,nrOfBatches-1 do
      local start = batch * batchSize + 1
      local myend = start + batchSize -1
      local batchInputs = trainData:sub(start, myend):cuda()
      local batchInputsNotCuda = targData:sub(start, myend)


      local function feval(params)
        gradParams:zero()
        local outputs = model:forward(batchInputs)
        local outputsNotCuda = torch.Tensor(BS,2,S1,S2)
--      print(outputs:size())
        outputsNotCuda:copy(outputs)
        local loss = criterion:forward(outputsNotCuda, batchInputsNotCuda)

        losses[batch + 1] = loss
        local dloss_doutput = criterion:backward(outputsNotCuda, batchInputsNotCuda)

--      print('here 1')

        local gradsCuda = torch.CudaTensor(BS,2,S1,S2)
        gradsCuda:copy(dloss_doutput)

        model:backward(batchInputs, gradsCuda)
--      print('here 2')
        return loss,gradParams
      end
--      optim.adadelta(feval, params, optimState)
      optim.sgd(feval, params, config)
    end
    local loss_avg = losses:sum() / nrOfBatches
    print('AVG LOSS: ' .. loss_avg)
    out:write(loss_avg)
    out:write(',')

  end
  out:write('\n')
end
out:close()
