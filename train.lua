-- training the model
--require('cutorch')
--require('cudnn') -- libcudnn missing
require('nn')
require('cunn')
require('optim')
require('paths')
require('sys')
require('gnuplot')

require 'os'
require 'img_msecrit.lua'
require 'synth_dataset'

local print_freq = 100
local sgd = false
local conv3d = false
local small = true

local epochX = torch.Tensor(10000)
local lossY = torch.Tensor(3,10000)

local data_dir = arg[1]
local trainingData = arg[2]
local targetData = arg[3]
local resDir = arg[4]
local batchSize = arg[5]
local epochs = arg[6]
local printF = arg[7]
local coarse2F = arg[8]
local normalize = arg[9]
local BS, channels, S1, S2 = 1, 3, 1, 1

if (conv3d) then
  require 'model/model_3d_conv_simple.lua'
elseif (small) then
  require 'model/model-small.lua'
else
  require 'model/model.lua'
end

require 'OpticalFlowCriterion'

function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end


----------------------------------------------------------------------
-- dataset 

--local trainData = torch.load(data_dir .. trainingData, 'ascii')/normalize
--local targData = torch.load(data_dir .. targetData, 'ascii'):sub(1,trainData:size(1))/normalize

local trainData, targData, GT, dataname = load_dataset(batchSize)
trainData = trainData - trainData:mean()

BS = targData:size(1)
S1 = targData:size(3)
S2 = targData:size(4)

local nrOfBatches = trainData:size(1)/batchSize
local size1 = trainData:size(3)
local size2 = trainData:size(4)

-- shuffling batches
for i = 1, batchSize * nrOfBatches do
  local j = math.random(i, batchSize*nrOfBatches)
  trainData[i], trainData[j] = trainData[j], trainData[i]
  targData[i], targData[j] = targData[j], targData[i]
  GT[i], GT[j] = GT[j], GT[i]
end

if conv3d then trainData = trainData:reshape(torch.LongStorage{BS,3,2,size1,size2}) end
----------------------------------------------------------------------
local out = nil
local readme = nil
local namedir = ''
local lr = 0.0001
local a_0 = 0.2
local a = a_0
----------------------------------------------------------------------
while lr < 0.001 do
  local losses = torch.Tensor(4)
  print('STARTING ALFA: ' .. a)
  print('STARTING LR: ' .. lr)

  epochX = torch.Tensor(10000)
  lossY = torch.Tensor(3,10000)
----------------------------------------------------------------------
-- Directory for results

  if printF then 
    local ind = 1
    local time = os.date("%X")
    local name = lr .. '-' .. a .. '-' .. BS .. '-' ..epochs ..'-' ..time
    namedir = 'results/'..resDir..'/'..name..'_' .. ind
    while (file_exists(namedir)) do
      ind = ind + 1
      namedir = 'results/'..resDir..'/'..name ..'_' .. ind
    end
    os.execute("mkdir " .. namedir) 
    os.execute("mkdir " .. namedir..'/images') 
    os.execute("mkdir " .. namedir..'/final') 
    os.execute("mkdir " .. namedir..'/final/images') 
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

  local model = create_model(channels, S1, S2, false, 2)
  print("Model:")
  print(model)
  readme:write(string.format('Model configuration:\n%s', model))

  print('Starting run: ' .. namedir .. ', using data: ' .. dataname)
----------------------------------------------------------------------
-- Trying to optimize memory

  opts = {inplace=true, mode='training'}
  optnet = require 'optnet'
  optnet.optimizeMemory(model, trainData:sub(1, batchSize):cuda(), opts)
  print('done optimization')

----------------------------------------------------------------------
-- CRITERION

  local criterion = nn.OpticalFlowCriterion(namedir, printF, a, normalize, GT)
----------------------------------------------------------------------
-- Optimization algorithm

  local params, gradParams = model:getParameters() -- to flatten all the matrices inside the model
  model.train = true

  local optimState = {}
  config = {
    learningRate = lr,
    weightDecay = 0,
    momentum = 0.9,
    learningRateDecay = 1e-3
  }
  if printF then
    readme:write(os.date("%x, %X \n"))
    readme:write('training data '..dataname ..'\n')
--      readme:write('alfa = ' .. a ..'\n')
    gnuplot.title('Loss, LR = ' .. lr .. ', alfa = ' .. a)
    readme:write('LR = ' .. config.learningRate ..'\n')
    readme:write('Alfa = ' .. a ..'\n')
    readme:write('momentum = ' .. config.momentum ..'\n')
    readme:write('batch size = ' .. batchSize ..'\n')
    readme:write('number of epochs = ' .. epochs ..'\n')
  end
----------------------------------------------------------------------
  local maxflow = 0
  local minflow = 0
  local lossAvg = 0
  local reg = 0
  local err = 0

----------------------------------------------------------------------
--  STARTING TRAINING!!!!

  for epoch=1,epochs do

    if (epoch == 1 or epoch % print_freq == 0) then print("STARTING EPOCH "..epoch) end
    local dloss_doutput = nil

    for batch = 0,nrOfBatches-1 do
      local start = batch * batchSize + 1
      local myend = start + batchSize -1
      local batchInputs = trainData:sub(start, myend):cuda()
      local batchInputsNotCuda = targData:sub(start, myend)

----------------------------------------------------------------------
--      Evalutaion function
      local function feval(params)
        gradParams:zero() --otherwise goes to hell :)
        local outputs = model:forward(batchInputs)
--          print(outputs:size())
        local outputsNotCuda = nil
        local gradsCuda = nil

        outputsNotCuda = torch.Tensor(BS,2,S1,S2)
        gradsCuda = torch.CudaTensor(BS,2,S1,S2)

        outputsNotCuda:copy(outputs)
        maxflow = outputsNotCuda:max()
        minflow = outputsNotCuda:min()

        if (epoch == 1 or epoch % print_freq == 0) then 
          print('max in flow: ' .. maxflow .. ', min in flow: ' .. minflow)
          print('<0 ' .. outputsNotCuda:lt(0):sum() ..', <-0.4 ' .. outputsNotCuda:lt(-0.4):sum() ..', <-0.8 ' .. outputsNotCuda:lt(-0.8):sum())
        end

        local loss, err, reg = criterion:forward(outputsNotCuda, batchInputsNotCuda)
        losses[batch + 1] = loss
        dloss_doutput = criterion:backward(outputsNotCuda, batchInputsNotCuda)
        gradsCuda:copy(dloss_doutput)

        model:backward(batchInputs, gradsCuda)

        return loss,gradParams
      end
----------------------------------------------------------------------
--      Choosing between SGD and ADADELTA

      if sgd then
        optim.sgd(feval, params, config)
      else
        optim.adadelta(feval, params, optimState)
      end

----------------------------------------------------------------------
    end
    lossAvg = losses:sum() / nrOfBatches

    if (epoch == 1 or epoch % print_freq == 0) then 
      print('AVG LOSS: ' .. lossAvg .. ' => ' .. err .. ' + ' .. reg)
      out:write(lossAvg .. ',')
    end
    epochX[epoch] = epoch
    lossY[1][epoch] = lossAvg
    lossY[2][epoch] = err
    lossY[3][epoch] = reg * a

--      annealing of alfa
    a = a_0 / (1+(epoch/(epochs/2)))
  end

  if (printF) then
    readme:write('Flow max = ' .. maxflow ..', min = ' .. minflow ..'\n')
    readme:write('Final loss = ' .. lossAvg ..' => ' .. err .. ' + ' .. reg .. '\n')
    readme:write('Stopped at: ' .. os.date("%X\n"))
    if sgd then 
      readme:write('Used optim: SGD\n')
    else
      readme:write('Used optim: Adadelta\n')
    end
    if conv3d then 
      readme:write('Used 3d convolution\n')
    else
      readme:write('Used normal convolution\n')
    end

    readme:close()
    out:write('\n')
    out:close()
  end

  gnuplot.pngfigure(namedir .. '/loss.png')
  gnuplot.title('Loss, LR = ' .. lr .. ', alfa = ' .. a)
  gnuplot.plot(
    {'Total error', epochX, lossY[1], '-'},
    {'Intensity error', epochX, lossY[2], '-'},
    {'TV error', epochX, lossY[3], '-'}
  )
  gnuplot.xlabel('time(epoch)')
  gnuplot.ylabel('E(x)')
  gnuplot.plotflush()
  print('Finished run ' .. namedir)
  lr = lr*10
end
--end