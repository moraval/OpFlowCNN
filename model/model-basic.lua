require 'torch'   -- torch
require 'image'   -- for image transforms
require 'cutorch'
require 'cunn'
require 'cudnn'

local nn = require 'nn' 
local nninit = require 'nninit'

--torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- reset weights - select type
local method = 'xavier'
local s1 = 64
local s2 = 64

function create_model(channels, size1, size2, batchSize)

  s1 = size1
  s2 = size2

  local function conv(nIn, nOut, k, s, p)
    local layer = nn.Sequential()
    layer:add(cudnn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
    layer = require('weight-init.lua')(layer, method)
    -- layer:add(cudnn.SpatialBatchNormalization(nOut))
    layer:add(nn.HardTanh(-10,10))
    -- layer:add(nn.Threshold(-10,-10,true))
    -- layer:add(nn.ReLU())
    -- layer:add(nn.LeakyReLU())
    return layer
  end

  local function deconv(nIn,nOut,k,s,p)
    local layer = nn.Sequential()
    layer:add(cudnn.SpatialFullConvolution(nIn,nOut,k,k,s,s,p,p)) 
    layer = require('weight-init.lua')(layer, method)
    -- layer:add(cudnn.SpatialBatchNormalization(nOut))
    layer:add(nn.HardTanh(-10,10))
    -- layer:add(nn.Threshold(-10,-10,true))
    -- layer:add(nn.ReLU())
    -- layer:add(nn.LeakyReLU())
    return layer
  end

  local function convTanh(nIn, nOut, k, s, p)
    local layer = nn.Sequential()
    layer:add(cudnn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
    layer = require('weight-init.lua')(layer, method)
    -- layer:add(cudnn.SpatialBatchNormalization(nOut))
    -- layer:add(nn.Tanh())
    layer:add(nn.HardTanh(-2,2))
    -- layer:add(nn.Threshold(-10,-10,true))
    return layer
  end
  
  local function combine(layer, shortcut)
    local inner_L = nn.Sequential()
    :add(nn.ConcatTable()
      :add(layer)
      :add(shortcut))
    :add(nn.CAddTable(true))
    return inner_L
  end

  model = nn.Sequential()

  -- 16, 32, 32
  local L1_chan = 32
  local L2_chan = 64
  local L3_chan = 128
  local L4_chan = 256/2
  local L5_chan = 512/2

  local p = 0.2

  -- if batchSize > 1 then
  --   model:add(nn.View(2*3,s1,s2))
  --   model:add(nn.Squeeze())
  -- else
  --   model:add(nn.View(1,2*3,s1,s2))
  -- end

  -- model:add(conv(6,L1_chan/2,9,1,4))
  -- model:add(conv(L1_chan/2,L1_chan,7,2,3))
  model:add(conv(6,L1_chan,7,2,3))
  model:add(conv(L1_chan,L1_chan,7,1,3))
  model:add(conv(L1_chan,L2_chan,5,2,2))
  model:add(conv(L2_chan,L2_chan,5,1,2))
  -- model:add(conv(L2_chan,L2_chan,5,1,2))
  -- model:add(conv(L3_chan,L2_chan,3,1,1))
  -- model:add(nn.SpatialDropout(p))

  local L = nn.Sequential()
  L:add(conv(L2_chan,L3_chan,3,2,1))
  L:add(deconv(L3_chan,L2_chan,4,2,1))

  -- conv, deconv + put there shortcut
  local shortcut = nn.Sequential()
  shortcut:add(nn.Identity())
  model:add(combine(L,shortcut))

  -- model:add(nn.SpatialDropout(p))
  -- model:add(conv(L2_chan,2,1,1,0))
  model:add(convTanh(L2_chan,2,1,1,0))
  
--  put on CUDA
  model:cuda()
  -- cudnn.convert(model, cudnn)
-----------------------------------------------------------------------
  return model
end
