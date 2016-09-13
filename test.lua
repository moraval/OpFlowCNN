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

local testData, dataname = load_dataset_test()
testData = testData - testData:mean()

BS = testData:size(1) / nrOfBatches
S1 = testData:size(3)
S2 = testData:size(4)
local size1 = testData:size(3)
local size2 = testData:size(4)

----------------------------------------------------------------------
local out = nil
local readme = nil
local namedir = ''

----------------------------------------------------------------------
-- Directory for results

  if printF then 
    local ind = 1
    local time = os.date("%X")
    local name = 'test_model-' .. modelname .. '-' .. dataname
    namedir = 'results/'..resDir..'/'..name
    os.execute("mkdir " .. namedir) 
    os.execute("mkdir " .. namedir..'/images') 
    readme = assert(io.open(namedir.."/readme", "w"))
  end


print('Loading model...')
local modelDir = '../model/crnn_demo/'
paths.dofile(paths.concat(modelDir, 'config.lua'))
local modelLoadPath = paths.concat(modelDir, 'crnn_demo_model.t7')
gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
local model, criterion = createModel(gConfig)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)
model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))

local imagePath = '../data/demo.png'
local img = loadAndResizeImage(imagePath)
local text, raw = recognizeImageLexiconFree(model, img)
print(string.format('Recognized text: %s (raw: %s)', text, raw))
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

        loss, err, reg = criterion:forward(outputsNotCuda, batchInputsNotCuda)
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
    if (epochs % print_freq == 0) then
      criterion = nn.OpticalFlowCriterion(namedir, printF, a, normalize, GT)
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
  print('Finished run ' .. namedir)
  lr = lr*10
end
--end