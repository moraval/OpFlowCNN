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
local mseCrit = false

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
local S1 = 1
local S2 = 1

function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end

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
    image.save('results/'..resDir..'/images/MSE-criterion/bigImg_'..printEpoch..'.png', bigImg)
  end
end

----------------------------------------------------------------------
-- dataset 

--local trainData = torch.load(data_dir .. trainingData, 'ascii')/normalize
--local targData = torch.load(data_dir .. targetData, 'ascii'):sub(1,trainData:size(1))/normalize

--local trainData, targData, GT = create_dataset(resDir)
local trainData, targData, GT = load_dataset(batchSize)

--trainData = trainData - trainData:mean()

BS = targData:size(1)
S1 = targData:size(3)
S2 = targData:size(4)

local nrOfBatches = trainData:size(1)/batchSize
local size1 = trainData:size(3)
local size2 = trainData:size(4)

--for i = 1, batchSize * nrOfBatches do
--  local j = math.random(i, batchSize*nrOfBatches)
--  trainData[i], trainData[j] = trainData[j], trainData[i]
--  targData[i], targData[j] = targData[j], targData[i]
--  GT[i], GT[j] = GT[j], GT[i]
--end

for a = 0.05,0.05,0.2 do
  local losses = torch.Tensor(4)
  print('STARTING ALFA: ' .. a)
----------------------------------------------------------------------
  local out = nil
  local readme = nil

----------------------------------------------------------------------
-- Directory for results

  local namedir = ''
  if printF then 
    local ind = 1
    namedir = 'results/'..resDir..'/'..alfa ..'_' .. ind
    while (file_exists(namedir)) do
      ind = ind + 1
      namedir = 'results/'..resDir..'/'..alfa ..'_' .. ind
    end
    os.execute("mkdir " .. namedir) 
    os.execute("mkdir " .. namedir..'/images') 
    os.execute("mkdir " .. namedir..'/flows') 
    ind = 1
    local losses_name = namedir .. '/losses.csv'
    out = assert(io.open(losses_name, "w")) -- open a file for serialization
    readme = assert(io.open(namedir.."/readme", "w"))
    out:write(a)
    out:write(',')
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
-- CRITERION

  local criterion = nil
  if mseCrit then
    criterion = nn.MSECriterion()
  else
    criterion = nn.OpticalFlowCriterion(namedir, channels, printF, a, normalize, GT)
  end
----------------------------------------------------------------------

  local params, gradParams = model:getParameters() -- to flatten all the matrices inside the model

  local optimState = {}
  config = {
    learningRate = 1e-2,
    momentum = 0.9
--    l2_decay:1e-3
  }
  if printF then
    readme:write("Model:\n")
--    readme:write(s)

    readme:write('alfa = ' .. a ..'\n')
    readme:write('LR = ' .. config.learningRate ..'\n')
    readme:write('momentum = ' .. config.momentum ..'\n')
    readme:write('batch size = ' .. batchSize ..'\n')
    readme:write('number of epochs = ' .. epochs ..'\n')
  end
----------------------------------------------------------------------
  local maxflow = 0
  local minflow = 0
  local lossAvg = 0
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
--      print(outputsNotCuda:size())
--      print(outputs:size())
        outputsNotCuda:copy(outputs)
        maxflow = outputsNotCuda:max()
        minflow = outputsNotCuda:min()

        if (epoch == 1 or epoch % print_freq == 0) then 
          print('max in flow: ' .. maxflow .. ', min in flow: ' .. minflow)
          print('under 0 ' .. outputsNotCuda:lt(0):sum())
          print('under -0.4 ' .. outputsNotCuda:lt(-0.4):sum())
          print('under -0.8 ' .. outputsNotCuda:lt(-0.8):sum())
        end

--        if (epoch == 1 or epoch % (5*print_freq) == 0) then 
--          print(outputsNotCuda[1]:lt(-0.4))
--        end

        local loss = nil
        if mseCrit then
          loss = criterion:forward(outputsNotCuda, GT)
        else
          loss = criterion:forward(outputsNotCuda, batchInputsNotCuda)
        end

        losses[batch + 1] = loss

        local dloss_doutput = nil
        if mseCrit then
          dloss_doutput = criterion:backward(outputsNotCuda, GT)
          create_img(outputsNotCuda[1], GT[1], dloss_doutput[1], batchInputsNotCuda[1], epoch)
        else
          dloss_doutput = criterion:backward(outputsNotCuda, batchInputsNotCuda)
        end


        local gradsCuda = torch.CudaTensor(BS,2,S1,S2)
        gradsCuda:copy(dloss_doutput)

        model:backward(batchInputs, gradsCuda)
        return loss,gradParams
      end
--      optim.adadelta(feval, params, optimState)
      optim.sgd(feval, params, config)
    end
    lossAvg = losses:sum() / nrOfBatches
    if (epoch == 1 or epoch % print_freq == 0) then print('AVG LOSS: ' .. lossAvg) end
    out:write(lossAvg)
    out:write(',')

  end
  readme:write('final loss = ' .. lossAvg ..'\n')
  out:write('\n')
end
readme:close()
out:close()

