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
require 'synth_dataset'
require 'save_results.lua'

local print_freq = 100
local sgd = false
local conv3d = false
local small = true
local conv_fc = false
local onlinelearning = true
local saveResults = true

local epochX = torch.Tensor(1000)
local lossY = torch.Tensor(3,1000)

local epochValX = torch.Tensor(1000)
local lossValY = torch.Tensor(3,1000)

local data_dir = arg[1]
local modelDir = arg[2]
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
elseif conv_fc then
  require 'model/model-conv_fc.lua'
else
  require 'model/model.lua'
end

require 'OpticalFlowCriterion'

----------------------------------------------------------------------
-- training dataset 

local trainData, targData, GT, dataname = load_dataset()
trainData = trainData - trainData:mean()

BS = targData:size(1)
S1 = targData:size(3)
S2 = targData:size(4)
nrOfBatches = trainData:size(1)
local size1 = trainData:size(3)
local size2 = trainData:size(4)

if conv3d then trainData = trainData:reshape(torch.LongStorage{BS,3,2,size1,size2}) end

trainData = trainData:cuda()

-- validation dataset 
local inValSet, outValSet, GTval, datanameVal = load_val_dataset()
local valSize = inValSet:size(1)
inValSet = inValSet - inValSet:mean()
inValSet = inValSet:cuda()

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

  epochX = torch.Tensor(1000)
  lossY = torch.Tensor(3,1000)
----------------------------------------------------------------------
-- Directory for results
  if printF then 

    local time = os.date("%d-%m-%X")
    local name = lr .. '-' .. a .. '-' .. BS .. '-' ..epochs ..'-' ..time

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

  local model = create_model(channels, S1, S2, false, batchSize)
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

  local criterion = nn.OpticalFlowCriterion(namedir, printF, a, normalize, GT, datasize)
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
  end
----------------------------------------------------------------------
  local maxflow = 0
  local minflow = 0
  local lossAvg = 0
  local regAvg = 0
  local errAvg = 0

----------------------------------------------------------------------
--  STARTING TRAINING

  for epoch=1,epochs do

    -- model.train = true
    model:training()

    if (epoch == 1 or epoch % print_freq == 0) then print("STARTING EPOCH "..epoch) end
    local dloss_doutput = nil

    for ind = 1,nrOfBatches do
--      Evaluation function
      local function feval(params)
        gradParams:zero()
        local outputs = model:forward(trainData:sub(ind, ind))

        if conv3d then outputs = outputs:view(1,2,16,16) end
        
        local outputsNotCuda = nil
        local gradsCuda = nil

        outputsNotCuda = torch.Tensor(1,2,S1,S2)
        gradsCuda = torch.CudaTensor(1,2,S1,S2)
        
        outputsNotCuda:copy(outputs)
        maxflow = outputsNotCuda:max()
        minflow = outputsNotCuda:min()

        local loss, err, reg = criterion:forward(outputsNotCuda, targData:sub(ind,ind))
        lossAvg = lossAvg + loss
        regAvg = regAvg + reg
        errAvg = errAvg + err

        if (epoch % print_freq == 0 and ind < 8) then
          local orig = torch.Tensor(3,S1,S2):copy(targData:sub(ind,ind,1, channels))
          save_results(trainNameDir, outputsNotCuda[1], image_estimate[1], orig, GT[ind], ind, epoch) 
        end

        dloss_doutput = criterion:backward(outputsNotCuda, targData:sub(ind,ind))
        gradsCuda:copy(dloss_doutput)

        model:backward(trainData:sub(ind, ind), gradsCuda)
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
    lossAvg = lossAvg / nrOfBatches
    errAvg = errAvg / nrOfBatches
    regAvg = regAvg / nrOfBatches

    if (epoch == 1 or epoch % print_freq == 0) then 
      print('TRAINING AVG LOSS: ' .. lossAvg .. ' => ' .. errAvg .. ' + ' .. regAvg)
      print('Max in flow: ' .. maxflow .. ', min in flow: ' .. minflow)
      out:write(lossAvg .. ',')

      -- validate
      model:evaluate()
      local lossValAvg, lossErrAvg, lossRegAvg = 0, 0, 0
      for v = 1,valSize do
        -- print(inValSet:sub(v,v):size())
        local outVal = model:forward(inValSet:sub(v,v))
        outValNotCuda = torch.Tensor(1,2,S1,S2):copy(outVal)
        local lossVal, lossEr, lossReg = criterion:forward(outValNotCuda, outValSet:sub(v,v))
        lossValAvg = lossValAvg + lossVal
        lossErrAvg = lossErrAvg + lossEr
        lossRegAvg = lossRegAvg + lossReg

        local orig = torch.Tensor(3,S1,S2):copy(outValSet:sub(v,v,1, channels))
        save_results(valNameDir, outValNotCuda[1], image_estimate[1], orig, GTval[v], v, epoch) 
      end
      print('VALIDATION AVG LOSS: ' .. lossValAvg/valSize .. ' => ' .. lossErrAvg/valSize .. ' + ' .. lossRegAvg/valSize*a)
      collectgarbage()

      epochValX[epoch] = epoch
      lossValY[1][epoch] = lossValAvg/valSize
      lossValY[2][epoch] = lossErrAvg/valSize
      lossValY[3][epoch] = lossRegAvg/valSize * a
      model.train = true
    end
    epochX[epoch] = epoch
    lossY[1][epoch] = lossAvg
    lossY[2][epoch] = errAvg
    lossY[3][epoch] = regAvg * a

    lossAvg, regAvg, errAvg = 0, 0, 0

--      annealing of alfa
    a = a_0 / (1+(epoch/(epochs/2)))
    if (epochs % print_freq == 0) then
      criterion = nn.OpticalFlowCriterion(namedir, printF, a, normalize, GT, datasize)
    end
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
  -- lr = lr*10

  if saveResults then torch.save(savePath, model) end
end
--end