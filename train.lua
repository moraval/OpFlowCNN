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

local print_freq = 10

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
local BS = 16
local S1 = 16
local S2 = 16

function create_img(flow, GT, gradient, imgs, epoch)
  if (epoch % print_freq == 0) or epoch == 1 then

    print(flow:size())
    local s1 = 2*16 + 2
    local s2 = 4*16 + 3*2

    local orig = imgs:sub(1,3)
    local target = imgs:sub(4,6)

    if epoch == 100 then print(flow) end

    local img = image.warp(target, flow, 'bilinear')
    local img_orig = image.warp(target, GT, 'bilinear')

    local printEpoch = string.format("%05d", epoch)

    local bigImg = torch.Tensor(1,s1,s2):fill(4)

    local flow1 = flow[1] + math.abs(torch.min(flow[1]))
    local flow2 = flow[2] + math.abs(torch.min(flow[2]))
    
    bigImg[1]:sub(1,16,1,16):copy(flow1)
    bigImg[1]:sub(19,34,1,16):copy(flow2)

    local GT1 = GT[1] + math.abs(torch.min(GT[1]))
    local GT2 = GT[2] + math.abs(torch.min(GT[2]))
    
    bigImg[1]:sub(1,16,19,34):copy(GT1)
    bigImg[1]:sub(19,34,19,34):copy(GT2)

    local gr1 = gradient[1]
    gr1 = gr1 + math.abs(torch.min(gr1))
    gr1 = gr1 * (1/torch.max(gr1))

    local gr2 = gradient[2]
    gr2 = gr2 + math.abs(torch.min(gr2))
    gr2 = gr2 * (1/torch.max(gr2))

    bigImg[1]:sub(1,16,37,52):copy(gr1*8)
    bigImg[1]:sub(19,34,37,52):copy(gr2*8)

    bigImg[1]:sub(1,16,55,70):copy(img[1]*8)
    bigImg[1]:sub(19,34,55,70):copy(orig[1]*8)

    bigImg = bigImg/8
    printA = 5
    image.save('results/'..resDir..'/images/5-MSEcriterion/bigImg_'..printEpoch..'.png', bigImg)
  end
end
----------------------------------------------------------------------
-- dataset 

--local trainData = torch.load(data_dir .. trainingData, 'ascii')/normalize
--local targData = torch.load(data_dir .. targetData, 'ascii'):sub(1,trainData:size(1))/normalize

--local trainData, targData, GT = create_dataset(resDir)
local trainData, targData, GT = load_dataset()

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

for a = 0.01,0.01,0.2 do
  local losses = torch.Tensor(4)
--  local print_A = math.ceil(a*10)
  local print_A = a*10

  print('STARTING ALFA: ' .. a)
  out:write(print_A)
  out:write(',')

----------------------------------------------------------------------
-- CRITERION

  local criterion = nn.OpticalFlowCriterion(resDir, channels, batchSize, printF, a, normalize, GT)
--  local criterion = nn.MSECriterion()
----------------------------------------------------------------------

  local params, gradParams = model:getParameters() -- to flatten all the matrices inside the model

  local optimState = {}
  config = {
    learningRate = 1e-2,
    momentum = 0.5
  }
----------------------------------------------------------------------
  local maxflow = 0
  local minflow = 0
  for epoch=1,epochs do

    if (epoch == 1 or epoch % print_freq == 0) then 
      print("STARTING EPOCH "..epoch) 
--      print('--------------------FLOW-----------------------')
--      print(outputsNotCuda)
--      print('max in flow: ' .. maxflow .. ', min in flow: ' .. minflow)
    end

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
        maxflow = outputsNotCuda:max()
        minflow = outputsNotCuda:min()
--      print('--------------------FLOW-----------------------')
--      print(outputsNotCuda)
      if (epoch == 1 or epoch % print_freq == 0) then print('max in flow: ' .. maxflow .. ', min in flow: ' .. minflow) end

        local loss = criterion:forward(outputsNotCuda, batchInputsNotCuda)
--        local loss = criterion:forward(outputsNotCuda, GT)

        losses[batch + 1] = loss

        local dloss_doutput = criterion:backward(outputsNotCuda, batchInputsNotCuda)
--        local dloss_doutput = criterion:backward(outputsNotCuda, GT)

--        create_img(outputsNotCuda[1], GT[1], dloss_doutput[1], batchInputsNotCuda[1], epoch)

        local gradsCuda = torch.CudaTensor(BS,2,S1,S2)
        gradsCuda:copy(dloss_doutput)

        model:backward(batchInputs, gradsCuda)
        return loss,gradParams
      end
--      optim.adadelta(feval, params, optimState)
      optim.sgd(feval, params, config)
    end
    local loss_avg = losses:sum() / nrOfBatches
    if (epoch == 1 or epoch % print_freq == 0) then print('AVG LOSS: ' .. loss_avg) end
--    print('AVG LOSS: ' .. loss_avg)
    out:write(loss_avg)
    out:write(',')

  end
  out:write('\n')
end
out:close()

