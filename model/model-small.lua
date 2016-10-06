require 'torch'   -- torch
require 'image'   -- for image transforms
require('cutorch')
require('cunn')
require 'cudnn'

local nn = require 'nn' 
local nninit = require 'nninit'

--torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- reset weights - select type
local method = 'flow'

function create_model(channels, size1, size2, batchSize)

  local s1 = 64
  local s2 = 64

  local function conv(nIn, nOut, k, s, p)
    local layer = nn.Sequential()
    layer:add(cudnn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
    layer = require('weight-init.lua')(layer, method)
    -- layer:add(cudnn.SpatialBatchNormalization(nOut))
    layer:add(nn.HardTanh(-10,10))
    return layer
  end

  local function deconv(nIn,nOut,k,s,p)
    local layer = nn.Sequential()
    layer:add(cudnn.SpatialFullConvolution(nIn,nOut,k,k,s,s,p,p))
    layer = require('weight-init.lua')(layer, method)
    -- layer:add(cudnn.SpatialBatchNormalization(nOut))
    layer:add(nn.HardTanh(-10,10))
    return layer
  end

  local function conv3d(nIn, nOut, kT, k, sT, s, pT, p)
    local layer = nn.Sequential()
    layer:add(cudnn.VolumetricConvolution(nIn, nOut, kT, k, k,sT,s,s,pT,p,p))
    layer = require('weight-init.lua')(layer, method)
    -- layer:add(nn.VolumetricBatchNormalization(nOut))
    layer:add(nn.HardTanh(-10,10))
    return layer
  end

local function combine(layer)
    local inner_L = nn.Sequential()
    :add(nn.ConcatTable()
      :add(layer)
      :add(nn.Identity()))
    :add(nn.CAddTable(true))
    return inner_L
  end
----------------------------------------------------------------------
-- layers inside layers - so that I can create the shortcuts (viz. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  -- cudnn.benchmark = true 
  -- cudnn.fastest = true

  model = nn.Sequential()

  local L1_chan = 32
  local L2_chan = 64
  local L3_chan = 128
  local L4_chan = 256
  local L5_chan = 512

  local p = 0.2

  if batchSize > 1 then
    model:add(nn.View(batchSize,1,6,s1,s2))
  else
    model:add(nn.View(1,1,6,s1,s2))
  end

  model:add(conv3d(1, L1_chan, 3, 3, 3, 1, 0, 1))
  model:add(conv3d(L1_chan, L1_chan, 1, 3, 1, 1, 0, 1))
  model:add(conv3d(L1_chan, 2, 1, 3, 1, 1, 0, 1))
  model:add(nn.VolumetricDropout(p))

  if batchSize > 1 then
    model:add(nn.View(2*2,s1,s2))
    model:add(nn.Squeeze())
  else
    model:add(nn.View(1,2*2,s1,s2))
  end

  -- model:add(conv(2*2, L1_chan, 11, 2, 5))
  model:add(conv(2*2, L1_chan, 15, 2, 7))
  model:add(conv(L1_chan, L1_chan, 11, 1, 5))
  model:add(conv(L1_chan, L2_chan, 11, 1, 5))
  model:add(nn.SpatialDropout(p))

  local L = nn.Sequential()
  L:add(conv(L2_chan, L2_chan, 7, 2, 3))
  L:add(conv(L2_chan, L3_chan, 7, 1, 3))
  L:add(nn.SpatialDropout(p))
  L:add(conv(L3_chan, L3_chan, 5, 1, 2))
  L:add(conv(L3_chan, L3_chan, 3, 1, 1))

  L:add(conv(L3_chan, L4_chan, 5, 2, 2))
  L:add(nn.SpatialDropout(p))
  L:add(conv(L4_chan, L3_chan, 3, 1, 1))
  
  L:add(deconv(L3_chan, L3_chan, 6, 2, 2))
  L:add(deconv(L3_chan, L2_chan, 8, 2, 3))

  model:add(combine(L))

  model:add(conv(L2_chan, L1_chan, 11, 1, 5))
  model:add(nn.SpatialDropout(p))
  model:add(conv(L1_chan, 2, 1, 1, 0))

--  reset weights
  model = require('weight-init.lua')(model, method)
--  put on CUDA
  model:cuda()
  -- cudnn.convert(model, cudnn)
-----------------------------------------------------------------------
  return model
end
