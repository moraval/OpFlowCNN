-- training the model
require('cutorch')
require('cudnn') -- libcudnn missing
require('nn')
require('cunn')
require('optim')
require('paths')
require('sys')
require('gnuplot')

require 'os'
require 'synth_dataset'
require 'save_results.lua'

local print_freq = 10
local epochsGraph = 500
local sgd = false
local conv3d = false
local small = true
local conv_fc = false
local saveResults = false

local epochX = torch.Tensor(epochsGraph)
local lossY = torch.Tensor(3,epochsGraph)

local epochValX = torch.Tensor(epochsGraph/print_freq)
local lossValY = torch.Tensor(3,epochsGraph/print_freq)

local data_dir = arg[1]
local modelDir = arg[2]
local targetData = arg[3]
local resDir = arg[4]
-- local batchSize = arg[5]
local batchSize = 1
local epochs = arg[6]
local printF = arg[7]
local coarse2F = arg[8]
local normalize = arg[9]
local BS, channels, S1, S2 = 1, 3, 1, 1

if (conv3d) then
  require 'model/model_3d_conv_simple.lua'
elseif (small) then
  require 'model/model-small.lua'
elseif conv_fc then
  require 'model/model-conv_fc.lua'
else
  require 'model/model.lua'
end

require 'OpticalFlowCriterion-GPU'

----------------------------------------------------------------------
-- training dataset 

local trainData, targData, GT, dataname = load_dataset()
trainData = trainData - trainData:mean()

BS = targData:size(1)
S1 = targData:size(3)
S2 = targData:size(4)
local size1 = trainData:size(3)
local size2 = trainData:size(4)

-- validation dataset 
local inValSet, outValSet, GTval, datanameVal = load_val_dataset()
local valSize = inValSet:size(1)
inValSet = inValSet - inValSet:mean()

trainData = trainData / torch.std(trainData)
inValSet = inValSet / torch.std(inValSet)

trainData = trainData:cuda()
targData = targData:cuda()
inValSet = inValSet:cuda()
outValSet = outValSet:cuda()
GT = GT:cuda()
GTval = GTval:cuda()

if conv3d then 
  trainData = trainData:reshape(torch.LongStorage{BS,1,6,size1,size2})
  inValSet = inValSet:reshape(torch.LongStorage{valSize,1,6,size1,size2})
end
----------------------------------------------------------------------
local out = nil
local readme = nil
local namedir = ''
local savePath = ''
local lr = 0.0001
local a_0 = 0.14
local a = a_0
----------------------------------------------------------------------
while lr < 0.001 do
  print('STARTING ALFA: ' .. a)
  print('STARTING LR: ' .. lr)

  epochX = torch.Tensor(epochsGraph)
  lossY = torch.Tensor(3,epochsGraph)
----------------------------------------------------------------------
-- Directory for results
  if printF then 

    local time = os.date("%d-%m-%X")
    local name = lr .. '-' .. a .. '-' .. BS .. '-' ..epochs ..'-' ..time
    
    local f=io.open('results/'..resDir,"r")
    local exists = f~=nil
    if exists then 
      io.close(f)
    else
      local newDir = 'results/'..resDir
      os.execute("mkdir " .. newDir)
    end

    savePath = 'models-learned/' .. name ..'.t7'
    namedir = 'results/'..resDir..'/'..name
    valNameDir = namedir .. '/val_img'
    trainNameDir = namedir .. '/train_img'
    trainFinNameDir = namedir .. '/train_fin_img'

    os.execute("mkdir " .. namedir) 
    os.execute("mkdir " .. valNameDir) 
    os.execute("mkdir " .. trainNameDir) 
    os.execute("mkdir " .. trainFinNameDir) 
    os.execute("mkdir " .. namedir..'/flows') 

    local losses_name = namedir .. '/losses.csv'
    out = assert(io.open(losses_name, "w")) -- open a file for serialization
    readme = assert(io.open(namedir.."/readme", "w"))
    out:write(a)
    out:write(',')
  end

----------------------------------------------------------------------
-- `create_model` is defined in model.lua, it returns the network model

  local model = create_model(channels, S1, S2, batchSize)
  print("Model:")
  print(model)
  readme:write(string.format('Model configuration:\n%s', model))

  print('Starting run: ' .. namedir .. ', using data: ' .. dataname)
----------------------------------------------------------------------
-- Trying to optimize memory

  opts = {inplace=true, mode='training'}
  optnet = require 'optnet'
  -- help = torch.CudaTensor(1,6,size1,size2):copy(trainData:sub(1, 1))
  optnet.optimizeMemory(model, trainData:sub(1, batchSize), opts)
  print('done optimization')

----------------------------------------------------------------------
-- CRITERION

  local criterion = nn.OpticalFlowCriterionGPU(namedir, printF, a, normalize, GT, datasize)
----------------------------------------------------------------------
-- Optimization algorithm

  local params, gradParams = model:getParameters() -- to flatten all the matrices inside the model

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
    readme:write('LR = ' .. config.learningRate ..'\n')
    readme:write('Alfa = ' .. a ..'\n')
    readme:write('momentum = ' .. config.momentum ..'\n')
    readme:write('batch size = ' .. batchSize ..'\n')
    readme:write('number of epochs = ' .. epochs ..'\n')

    readme:flush()
  end
----------------------------------------------------------------------
  local maxflow = -10
  local minflow = 10
  local lossAvg = 0
  local regAvg = 0
  local errAvg = 0

  local memoryBatches = BS/batchSize
----------------------------------------------------------------------
--  STARTING TRAINING

  for epoch=1,epochs do

    lossAvg, regAvg, errAvg = 0, 0, 0
    model.train = true
    model:training()

    if (epoch == 1 or epoch % print_freq == 0) then 
      print('Max in flow: ' .. maxflow .. ', min in flow: ' .. minflow ..'\n')
      print("STARTING EPOCH "..epoch)
      print(os.date("%X \n"))
      maxflow = -10
      minflow = 10
    end
    local dloss_doutput = nil

    for m = 1, memoryBatches do
      local offset = (m-1)*batchSize + 1
      local offset_end = m*batchSize

--      Evaluation function
      local function feval(params)
        gradParams:zero()
        local outputs = model:forward(trainData:sub(offset,offset_end))

        maxflow = math.max(outputs:max(),maxflow)
        minflow = math.min(outputs:min(),minflow)

        local loss, err, reg = criterion:forward(outputs, targData:sub(offset,offset_end))
        lossAvg = lossAvg + loss
        regAvg = regAvg + reg
        errAvg = errAvg + err

        dloss_doutput = criterion:backward(outputs, targData:sub(offset,offset_end))

        if ((epoch == 1 or epoch % print_freq == 0) and m % 32 == 0) then 
            local orig = torch.Tensor(3,S1,S2):copy(targData:sub(offset,offset,1, channels))
            local dloss_CPU = torch.Tensor(2,S1,S2):copy(dloss_doutput[1])
            save_results(trainNameDir, outputs[1], image_estimate[1], orig, GT[offset], offset, epoch, true, dloss_CPU) 
        end

        model:backward(trainData:sub(offset,offset_end), dloss_doutput)
        return loss,gradParams
      end
  ----------------------------------------------------------------------
  --      Choosing between SGD and ADADELTA

      if sgd then
        optim.sgd(feval, params, config)
      else
        optim.adadelta(feval, params, optimState)
      end
    end
-------------------------------------------------------------------------
    lossAvg = lossAvg / memoryBatches
    errAvg = errAvg / memoryBatches
    regAvg = regAvg / memoryBatches

    if (epoch == 1 or epoch % print_freq == 0) then 
      print('TRAINING AVG LOSS: ' .. lossAvg .. ' => ' .. errAvg .. ' + ' .. regAvg)
      print(os.date("%X \n"))
      out:write(lossAvg .. ',')
      out:flush()

      -- validate
      model:evaluate()
      local lossValAvg, lossErrAvg, lossRegAvg = 0, 0, 0
      local memoryBatchesVal = valSize/batchSize

      for m = 1, memoryBatchesVal do 
        local offset = (m-1)*batchSize + 1
        local offset_end = m*batchSize

        local outVal = model:forward(inValSet:sub(offset,offset_end))
        local lossVal, lossEr, lossReg = criterion:forward(outVal, outValSet:sub(offset,offset_end))
        lossValAvg = lossValAvg + lossVal
        lossErrAvg = lossErrAvg + lossEr
        lossRegAvg = lossRegAvg + lossReg

        if (m % 8 == 0) then
          local orig = torch.Tensor(3,S1,S2):copy(outValSet:sub(offset,offset,1,channels))
          save_results(valNameDir, outVal[1], image_estimate[1], orig, GTval[offset], offset, epoch, false) 
        end
      end
      print('VALIDATION AVG LOSS: ' .. lossValAvg/valSize .. ' => ' .. lossErrAvg/valSize .. ' + ' .. lossRegAvg/valSize*a)
      -- collectgarbage()

      local writeEpoch = math.ceil(epoch/print_freq)
      epochValX[writeEpoch] = writeEpoch
      lossValY[1][writeEpoch] = lossValAvg/valSize
      lossValY[2][writeEpoch] = lossErrAvg/valSize
      lossValY[3][writeEpoch] = lossRegAvg/valSize * a
      model.train = true
    end

    -- print(os.date("%X \n"))
    epochX[epoch] = epoch
    lossY[1][epoch] = lossAvg
    lossY[2][epoch] = errAvg
    lossY[3][epoch] = regAvg * a

--      annealing of alfa
    a = a_0 / (1+(epoch/(epochs)))
    if (epochs % print_freq == 0) then
      criterion = nn.OpticalFlowCriterionGPU(namedir, printF, a, normalize, GT, datasize)
    end
  end

  if (printF) then
    readme:write('Flow max = ' .. maxflow ..', min = ' .. minflow ..'\n')
    readme:write('Final loss = ' .. lossAvg ..' => ' .. errAvg .. ' + ' .. regAvg .. '\n')
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

  print('Ended run at ' .. os.date("%X \n"))

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
    
  gnuplot.pngfigure(namedir .. '/lossVal.png')
  gnuplot.title('Validation loss, LR = ' .. lr .. ', alfa = ' .. a)
  gnuplot.plot(
    {'Total error', epochValX, lossValY[1], '-'},
    {'Intensity error', epochValX, lossValY[2], '-'},
    {'TV error', epochValX, lossValY[3], '-'}
  )
  gnuplot.xlabel('time(epoch)')
  gnuplot.ylabel('E(x)')
  gnuplot.plotflush()

  print('Finished run ' .. namedir)
  lr = lr*10

  if saveResults then torch.save(savePath, model) end
end
--end