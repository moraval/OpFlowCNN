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

local opt.print_freq = 10
local opt.epochs = 500
local sgd = false
local conv3d = false
local model = 1
-- local small = false
-- local only_conv = false
local saveResults = false
local printF = true
-- local batchSize = 1
local BS, channels, S1, S2 = 1, 3, 1, 1
local opt.dataSize = 1
local opt.valSize = 1
local loadModel = false

local cmd = torch.CmdLine()
cmd:option("-data_dir",'',"")
cmd:option("-result_dir",'',"")
cmd:option("-sgd",0,"Sgd or adadelta optimization algorithm")
cmd:option("-model",0,"")
cmd:option("-batchSize",0,"")
cmd:option("-epochs",0,"")
cmd:option("-print_freq",0,"")
cmd:option("-print_img",0,"")
cmd:option("-save_model",0,"")
cmd:option("-dataSize",0,"")
cmd:option("-valSize",0,"")
cmd:option("-load_model",0,"")
cmd:option("-modelname",0,"")
cmd:text()
local opt = cmd:parse(arg or {})

-- local data_dir = arg[1]
-- local resDir = arg[2]
sgd = (opt.sgd == 1)
-- model = tonumber(arg[4])
-- only_conv = (tonumber(arg[4]) == 1)
-- small = (tonumber(arg[5]) == 1)
-- batchSize = tonumber(arg[5])
-- opt.epochs = tonumber(arg[6])
-- opt.print_freq = tonumber(arg[7])
printF = (opt.print_img == 1)
saveResults = (opt.save_model == 1)
-- opt.dataSize = tonumber(arg[10])
-- opt.valSize = tonumber(arg[11])
loadModel = (opt.load_model == 1)
-- local opt.modelname = arg[13]

local epochX = torch.Tensor(opt.epochs)
local lossY = torch.Tensor(3,opt.epochs)

local epochValX = torch.Tensor(opt.epochs/opt.print_freq)
local lossValY = torch.Tensor(3,opt.epochs/opt.print_freq)

if (opt.model == 1) then
  require 'model/model-small-only-conv.lua'
elseif (opt.model == 2) then
  require 'model/model-small.lua'
elseif (opt.model == 3) then
  require 'model/model-basic.lua'
elseif (opt.model == 4) then
  require 'model/model-parallel.lua'
else
  require 'model/model.lua'
end

require 'OpticalFlowCriterion-GPU'

----------------------------------------------------------------------
-- training dataset 

local trainData, targData, GT, dataname = load_dataset(opt.dataSize)
trainData = trainData - trainData:mean()

BS = targData:size(1)
S1 = targData:size(3)
S2 = targData:size(4)
local size1 = trainData:size(3)
local size2 = trainData:size(4)

-- validation dataset 
local inValSet, outValSet, GTval, datanameVal = load_val_dataset(opt.valSize)
opt.valSize = inValSet:size(1)
inValSet = inValSet - inValSet:mean()

-- trainData = trainData / torch.std(trainData)
-- inValSet = inValSet / torch.std(inValSet)

trainData = trainData:cuda()
targData = targData:cuda()
inValSet = inValSet:cuda()
outValSet = outValSet:cuda()
GT = GT:cuda()
GTval = GTval:cuda()

if conv3d then 
  trainData = trainData:reshape(torch.LongStorage{BS,1,6,size1,size2})
  inValSet = inValSet:reshape(torch.LongStorage{opt.valSize,1,6,size1,size2})
end
----------------------------------------------------------------------
local out = nil
local readme = nil
local namedir = ''
local savePath = ''
local lr_0 = 0.001
local lr = lr_0
local a_0 = 0.05
local a = a_0
----------------------------------------------------------------------
while lr < 0.01 do
  print('STARTING ALFA: ' .. a)
  print('STARTING LR: ' .. lr)

  epochX = torch.Tensor(opt.epochs)
  lossY = torch.Tensor(3,opt.epochs)
----------------------------------------------------------------------
-- Directory for results
  if printF then 

    local time = os.date("%d-%m-%X")
    local name = lr .. '-' .. a .. '-' .. BS .. '-' ..opt.epochs ..'-' ..time
    
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
    procNameDir = namedir .. '/epoch_img'
    trainNameDir = namedir .. '/train_img'
    trainFinNameDir = namedir .. '/train_fin_img'

    os.execute("mkdir " .. namedir) 
    os.execute("mkdir " .. valNameDir) 
    os.execute("mkdir " .. trainNameDir) 
    os.execute("mkdir " .. procNameDir) 
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

  local model = create_model(channels, S1, S2, opt.batchSize)

  if (loadModel) then
    -- local opt.modelname = '0.001-0.005-1500-300-29-09-20:53:08.t7'
    local modelDir = 'models-learned/'
    local modelLoadPath = paths.concat(modelDir, opt.modelname)
    local model = torch.load(modelLoadPath)
  end
  print("Model:")
  print(model)
  readme:write(string.format('Model configuration:\n%s', model))

  print('Starting run: ' .. namedir .. ', using data: ' .. dataname)
----------------------------------------------------------------------
-- Trying to optimize memory

  opts = {inplace=true, mode='training'}
  optnet = require 'optnet'
  -- help = torch.CudaTensor(1,6,size1,size2):copy(trainData:sub(1, 1))
  optnet.optimizeMemory(model, trainData:sub(1, opt.batchSize), opts)
  print('done optimization')

----------------------------------------------------------------------
-- CRITERION

  local criterion = nn.OpticalFlowCriterionGPU(a, normalize, targData[1][1])
----------------------------------------------------------------------
-- Optimization algorithm

  local params, gradParams = model:getParameters() -- to flatten all the matrices inside the model

  local optimState = {}
  config = {
    learningRate = lr,
    weightDecay = 0,
    momentum = 0.5,
    learningRateDecay = 1e-3
  }
  if printF then
    readme:write(os.date("%x, %X \n"))
    readme:write('training data '..dataname ..'\n')
    readme:write('using criterion: OF\n')
    readme:write('LR = ' .. config.learningRate ..'\n')
    readme:write('Alfa = ' .. a ..'\n')
    readme:write('momentum = ' .. config.momentum ..'\n')
    readme:write('batch size = ' .. opt.batchSize ..'\n')
    readme:write('number of opt.epochs = ' .. opt.epochs ..'\n')

    readme:flush()
  end
----------------------------------------------------------------------
  local maxflow = -10
  local minflow = 10
  local lossAvg = 0
  local regAvg = 0
  local errAvg = 0

  local memoryBatches = BS/opt.batchSize
----------------------------------------------------------------------
--  STARTING TRAINING

  for epoch=1,opt.epochs do

    lossAvg, regAvg, errAvg = 0, 0, 0
    model.train = true
    model:training()

    if (epoch == 1 or epoch % opt.print_freq == 0) then 
      print('Max in flow: ' .. maxflow .. ', min in flow: ' .. minflow ..'\n')
      print("STARTING EPOCH "..epoch)
      print(os.date("%X \n"))
      maxflow = -10
      minflow = 10
    end
    local dloss_doutput = nil

    -- heretimer = torch.Timer()

    for m = 1, memoryBatches do
      local offset = (m-1)*opt.batchSize + 1
      local offset_end = m*opt.batchSize

--      Evaluation function
      local function feval(params)
        gradParams:zero()

        -- heretimer:reset()

        outputs = model:forward(trainData:sub(offset,offset_end))
        -- print(outputs:size())
        -- print(targData:sub(offset,offset_end):size())
        -- print('Time in forward ' .. heretimer:time().real ..'seconds')
        -- heretimer:reset()

        -- print(outputs:size())
        maxflow = math.max(outputs:max(),maxflow)
        minflow = math.min(outputs:min(),minflow)

        local loss, err, reg = criterion:forward(outputs, targData:sub(offset,offset_end))
        lossAvg = lossAvg + loss
        regAvg = regAvg + reg
        errAvg = errAvg + err

        dloss_doutput = criterion:backward(outputs, targData:sub(offset,offset_end))

        -- print('Time in OF ' .. heretimer:time().real ..'seconds')
        -- heretimer:reset()

        if ((epoch == 1 or epoch % opt.print_freq == 0) and m % 28 == 0) then 
            local orig = torch.Tensor(3,S1,S2):copy(targData:sub(offset,offset,1, channels))
            local dloss_CPU = torch.Tensor(2,S1,S2):copy(dloss_doutput[1])
            save_results(trainNameDir, outputs[1], image_estimate[1], orig, GT[offset], offset, epoch, false, false, dloss_CPU) 
        end
        if ((epoch == 1 or epoch % opt.print_freq/2 == 0) and m % 56 == 0) then 
            local orig = torch.Tensor(3,S1,S2):copy(targData:sub(offset,offset,1, channels))
            save_results(procNameDir, outputs[1], image_estimate[1], orig, GT[offset], offset, epoch, false, true) 
        end

        model:backward(trainData:sub(offset,offset_end), dloss_doutput)

        -- print('Time in backward + save ' .. heretimer:time().real ..'seconds')
        -- heretimer:reset()
        -- No, this was not the problem
        -- gradParams = gradParams/batchSize
        -- print(gradParams:size())
        -- print(gradParams:sub(1,100):view(10,10))
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

    -- local orig = torch.Tensor(3,S1,S2):copy(targData:sub(1,1,1,channels))
    -- save_results(procNameDir, outputs[1], image_estimate[1], orig, GT[1], 1, epoch, false, true) 
-------------------------------------------------------------------------
    lossAvg = lossAvg / memoryBatches
    errAvg = errAvg / memoryBatches
    regAvg = regAvg / memoryBatches

    if (epoch == 1 or epoch % opt.print_freq == 0) then 
      print('TRAINING AVG LOSS: ' .. lossAvg .. ' => ' .. errAvg .. ' + ' .. regAvg)
      print(os.date("%X \n"))
      out:write(lossAvg .. ',')
      out:flush()

      -- validate
      model:evaluate()
      local lossValAvg, lossErrAvg, lossRegAvg = 0, 0, 0
      local memoryBatchesVal = opt.valSize/opt.batchSize

      for m = 1, memoryBatchesVal do 
        local offset = (m-1)*opt.batchSize + 1
        local offset_end = m*opt.batchSize

        local outVal = model:forward(inValSet:sub(offset, offset_end))
        local lossVal, lossEr, lossReg = criterion:forward(outVal, outValSet:sub(offset,offset_end))
        lossValAvg = lossValAvg + lossVal
        lossErrAvg = lossErrAvg + lossEr
        lossRegAvg = lossRegAvg + lossReg

        if (m % 8 == 0) then
          local orig = torch.Tensor(3,S1,S2):copy(outValSet:sub(offset,offset,1,channels))
          save_results(valNameDir, outVal[1], image_estimate[1], orig, GTval[offset], offset, epoch, false, false) 
        end
      end 
      print('VALIDATION AVG LOSS: ' .. lossValAvg/opt.valSize .. ' => ' .. lossErrAvg/opt.valSize .. ' + ' .. lossRegAvg/opt.valSize*a)
      -- collectgarbage()

      local writeEpoch = math.ceil(epoch/opt.print_freq)
      epochValX[writeEpoch] = writeEpoch
      lossValY[1][writeEpoch] = lossValAvg/opt.valSize
      lossValY[2][writeEpoch] = lossErrAvg/opt.valSize
      lossValY[3][writeEpoch] = lossRegAvg/opt.valSize * a
      model.train = true
    end

    -- print(os.date("%X \n"))
    epochX[epoch] = epoch
    lossY[1][epoch] = lossAvg
    lossY[2][epoch] = errAvg
    lossY[3][epoch] = regAvg * a

--      annealing of alfa
    -- a = a_0 / (1+(epoch/(opt.epochs)))
    -- if (opt.epochs % opt.print_freq == 0) then
    --   criterion = nn.OpticalFlowCriterionGPU(a, normalize, targData[1][1])
    -- end
    -- annealing of LR
    -- if (opt.epochs % opt.print_freq == 0) then
    --   lr = lr_0 / (1+(epoch/(opt.epochs)))
    -- end
  end

  if (printF) then
    readme:write('Flow max = ' .. maxflow ..', min = ' .. minflow ..'\n')
    print('Flow max = ' .. maxflow ..', min = ' .. minflow)
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
    readme:write(string.format('Model configuration:\n%s', outputs))
    -- print(outputs)

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