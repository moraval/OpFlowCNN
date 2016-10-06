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

  -- s1 = size1
  -- s2 = size2

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

  local function conv3d(nIn, nOut, kT, k, sT, s, pT, p)
    local layer = nn.Sequential()
    layer:add(nn.VolumetricConvolution(nIn, nOut, kT, k, k,sT,s,s,pT,p,p))
    layer = require('weight-init.lua')(layer, method)
    -- layer:add(nn.VolumetricBatchNormalization(nOut))
    layer:add(nn.HardTanh(-10,10))
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
  local L0_chan = 16
  local L1_chan = 32
  local L2_chan = 64
  local L3_chan = 128
  local L4_chan = 256/2
  local L5_chan = 512/2

  local p = 0.2

  if batchSize > 1 then
    model:add(nn.View(batchSize,1,6,s1,s2))
  else
    model:add(nn.View(1,1,6,s1,s2))
  end

  model:add(conv3d(1, L0_chan, 3, 7, 3, 2, 0, 3))
  model:add(conv3d(L0_chan, L0_chan, 1, 7, 1, 1, 0, 3))
  -- model:add(conv3d(1, L1_chan, 3, 7, 3, 1, 0, 3))
  model:add(conv3d(L0_chan, L1_chan, 1, 5, 1, 2, 0, 2))
  model:add(conv3d(L1_chan, L1_chan, 1, 5, 1, 1, 0, 2))

  local L = nn.Sequential()
  L:add(conv3d(L1_chan, L2_chan, 1, 3, 1, 2, 0, 1))
  if batchSize > 1 then
    L:add(nn.View(L2_chan*2,s1/8,s2/8))
    L:add(nn.Squeeze())
  else
    L:add(nn.View(1,L2_chan*2,s1/8,s2/8))
  end
  L:add(deconv(L2_chan*2,L1_chan*2,4,2,1))

  local shortcut = nn.Sequential()
  shortcut:add(nn.Identity())
  shortcut:add(nn.View(1,1,L1_chan*2,s1/4,s2/4))

  -- -- conv, deconv + put there shortcut
  model:add(combine(L,shortcut))

  -- if batchSize > 1 then
  --   model:add(nn.View(L1_chan*2,s1/4,s2/4))
  --   model:add(nn.Squeeze())
  -- else
  --   model:add(nn.View(1,L1_chan*2,s1/4,s2/4))
  -- end
  -- put to U,V
  model:add(convTanh(L1_chan*2,2,1,1,0))
  
--  put on CUDA
  model:cuda()
  -- cudnn.convert(model, cudnn)
-----------------------------------------------------------------------
  return model
end
