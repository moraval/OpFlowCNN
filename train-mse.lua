-- training the model
require('cutorch')
require('cudnn') -- libcudnn missing
require('nn')
require('cunn')
require('optim')
require('paths')
require('sys')
require('gnuplot')
require 'image'

require 'os'
require 'synth_dataset'
require 'save_results.lua'

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

local sgd = (opt.sgd == 1)
local printF = (opt.print_img == 1)
local saveResults = (opt.save_model == 1)
local loadModel = (opt.load_model == 1)
local BS, channels, S1, S2 = 1, 3, 1, 1

local epochX = torch.Tensor(opt.epochs)
local lossY = torch.Tensor(opt.epochs)

local epochValX = torch.Tensor(opt.epochs/opt.print_freq)
local lossValY = torch.Tensor(opt.epochs/opt.print_freq)

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

----------------------------------------------------------------------
-- dataset 
local trainData, targData, GT, dataname = load_dataset(opt.dataSize)
trainData = trainData - trainData:mean()

BS = targData:size(1)
S1 = targData:size(3)
S2 = targData:size(4)
local size1 = trainData:size(3)
local size2 = trainData:size(4)

-- validation dataset 
local inValSet, outValSet, GTval, datanameVal = load_val_dataset(opt.dataSize, opt.valSize)
local valSize = inValSet:size(1)
inValSet = inValSet - inValSet:mean()

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
local lr = 0.1
-- local a_0 = 0.2
-- local a = a_0
----------------------------------------------------------------------
while lr < 1 do
  print('STARTING LR: ' .. lr)

  epochX = torch.Tensor(opt.epochs)
  lossY = torch.Tensor(opt.epochs)
----------------------------------------------------------------------
-- Directory for results

  if printF then 
    local time = os.date("%d-%m-%X")
    local name = lr .. '-' .. BS .. '-' ..opt.epochs ..'-' ..time

    local f=io.open('results/'..opt.result_dir,"r")
    local exists = f~=nil
    if exists then 
      io.close(f)
    else
      local newDir = 'results/'..opt.result_dir
      os.execute("mkdir " .. newDir)
    end

    savePath = 'models-learned/' .. name ..'.t7'
    namedir = 'results/'..opt.result_dir..'/'..name
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
    out:write(',')
  end

  ----------------------------------------------------------------------
  -- `create_model` is defined in model.lua, it returns the network model

  local model = create_model(channels, S1, S2, opt.batchSize)
  print("Model:")
  print(model)
  readme:write(string.format('Model configuration:\n%s', model))

  print('Starting run: ' .. namedir .. ', using data: ' .. dataname)
  ----------------------------------------------------------------------
  -- Trying to optimize memory

  opts = {inplace=true, mode='training'}
  optnet = require 'optnet'
  -- help = torch.CudaTensor(opt.batchSize,6,size1,size2):copy(trainData:sub(1, opt.batchSize))
  -- help = help:reshape(torch.LongStorage{opt.batchSize,1,6,size1,size2})
  print(trainData:sub(1, opt.batchSize):size())
  optnet.optimizeMemory(model, trainData:sub(1, opt.batchSize), opts)
  print('done optimization')

  ----------------------------------------------------------------------
  -- CRITERION

  -- local criterion = nn.MSECriterion()
  local criterion = nn.AbsCriterion()
  criterion = criterion:cuda()
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
    readme:write('using criterion: MSE\n')
    -- readme:write('alfa = ' .. a ..'\n')
    gnuplot.title('Loss, LR = ' .. lr)
    readme:write('LR = ' .. config.learningRate ..'\n')
    -- readme:write('Alfa = ' .. a ..'\n')
    readme:write('momentum = ' .. config.momentum ..'\n')
    readme:write('batch size = ' .. opt.batchSize ..'\n')
    readme:write('number of opt.epochs = ' .. opt.epochs ..'\n')
  end
  ----------------------------------------------------------------------
  local maxflow = -10
  local minflow = 10
  local lossAvg = 0
  local reg = 0
  local err = 0

  -- local memoryBatches = BS/32
  local memoryBatches = BS/opt.batchSize

----------------------------------------------------------------------
  --  STARTING TRAINING!!!!
  for epoch=1,opt.epochs do

    lossAvg, regAvg, errAvg = 0, 0, 0
    model.train = true
    model:training()

    if (epoch == 1 or epoch % 10 == 0) then 
      print('Max in flow: ' .. maxflow .. ', min in flow: ' .. minflow ..'\n')
      print("STARTING EPOCH "..epoch)
      print(os.date("%X \n"))
      maxflow = -10
      minflow = 10
    end
    local dloss_doutput = nil

    for m = 1, memoryBatches do
      local offset = (m-1)*opt.batchSize

      --  Evaluation function
      local function feval(params)
        gradParams:zero() --otherwise goes to hell :)
        -- help = torch.CudaTensor(opt.batchSize,6,size1,size2):copy(trainData:sub(opt.batchSize*(m-1)+1,opt.batchSize*m))
        -- help = help:reshape(torch.LongStorage{opt.batchSize,1,6,size1,size2})
        
        local outputs = model:forward(trainData:sub(opt.batchSize*(m-1)+1,opt.batchSize*m))
        
        maxflow = math.max(outputs:max(),maxflow)
        minflow = math.min(outputs:min(),minflow)

        local loss = criterion:forward(outputs, GT:sub(opt.batchSize*(m-1)+1,opt.batchSize*m))
        lossAvg = lossAvg + loss

        dloss_doutput = criterion:backward(outputs, GT:sub(opt.batchSize*(m-1)+1,opt.batchSize*m))

        if ((epoch == 1 or epoch % opt.print_freq == 0) and m % 8 == 0) then 
          local orig = torch.Tensor(3,S1,S2):copy(targData:sub(offset+1,offset+1,1, channels))
          local targ = torch.Tensor(3,S1,S2):copy(targData:sub(offset+1,offset+1,4, channels*2))
          local output_CPU = torch.Tensor(2,S1,S2):copy(outputs[1])
          local img_estimate = image.warp(targ,output_CPU,'bilinear')
          local dloss_CPU = torch.Tensor(2,S1,S2):copy(dloss_doutput[1])
          save_results(trainNameDir, outputs[1], img_estimate, orig, GT[offset+1], 1+offset, epoch, false) 
          -- save_results(trainNameDir, outputs[1], img_estimate, orig, GT[offset+1], 1+offset, epoch, true, dloss_CPU) 
        end
        -- gradsCuda:copy(dloss_doutput)

        model:backward(trainData:sub(opt.batchSize*(m-1)+1,opt.batchSize*m), dloss_doutput)
        -- model:backward(help, dloss_doutput)
        return loss,gradParams
      end
      ----------------------------------------------------------------------
      -- Choosing between SGD and ADADELTA
      if sgd then
        optim.sgd(feval, params, config)
      else
        optim.adadelta(feval, params, optimState)
      end
    end
  ----------------------------------------------------------------------
    lossAvg = lossAvg / memoryBatches

    if (epoch == 1 or epoch % opt.print_freq == 0) then 
      print('TRAINING AVG LOSS: ' .. lossAvg)
      out:write(lossAvg .. ',')

      -- validate
      model:evaluate()
      local lossValAvg = 0
      local memoryBatchesVal = valSize/opt.batchSize

      for m = 1, memoryBatchesVal do
        
        local offset = (m-1)*opt.batchSize

        local outVal = model:forward(inValSet:sub(opt.batchSize*(m-1)+1,opt.batchSize*m))
        local lossVal = criterion:forward(outVal, GTval:sub(offset+1,opt.batchSize*m))
        lossValAvg = lossValAvg + lossVal

        if (m % 4 == 0) then
          local orig = torch.Tensor(3,S1,S2):copy(outValSet:sub(offset+1,offset+1,1,channels))
          local targ = torch.Tensor(3,S1,S2):copy(outValSet:sub(offset+1,offset+1,4,channels*2))
          local output_CPU = torch.Tensor(2,S1,S2):copy(outVal[1])
          local img_estimate = image.warp(targ,output_CPU,'bilinear')
          save_results(valNameDir, outVal[1], img_estimate, orig, GT[offset+1], offset+1, epoch, false)
        end
      end
      print('VALIDATION AVG LOSS: ' .. lossValAvg/valSize)
      -- collectgarbage()

      local writeEpoch = math.ceil(epoch/opt.print_freq)
      epochValX[writeEpoch] = writeEpoch
      lossValY[writeEpoch] = lossValAvg/valSize
      model.train = true                                              
    end

    epochX[epoch] = epoch
    lossY[epoch] = lossAvg

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

  gnuplot.pngfigure(namedir .. '/lossTrain.png')
  gnuplot.title('Loss, LR = ' .. lr)
  gnuplot.plot({'Total error', epochX, lossY, '-'})
  gnuplot.xlabel('time(epoch)')
  gnuplot.ylabel('E(x)')
  gnuplot.plotflush()

  gnuplot.pngfigure(namedir .. '/lossVal.png')
  gnuplot.title('Validation loss, LR = ' .. lr)
  gnuplot.plot({'Total error', epochValX, lossValY, '-'})
  gnuplot.xlabel('time(epoch)')
  gnuplot.ylabel('E(x)')
  gnuplot.plotflush()

  print('Finished run ' .. namedir)
  lr = lr*10
end