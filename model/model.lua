require 'torch'   -- torch
require 'image'   -- for image transforms
require('cutorch')
require('cunn')

local nn = require 'nn' 

local function BNInit(name)
  for k,v in pairs(model:findModules(name)) do
    v.weight:fill(1)
    v.bias:zero()
  end
end
--torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- reset weights - select type
local method = 'flow'

function create_model(channels, size1, size2)

  local function conv(nIn, nOut, k, s, p)
    local layer = nn.Sequential()
    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
    layer = require('weight-init')(layer, method)
    layer:add(nn.SpatialBatchNormalization(nOut))
--    layer:add(nn.ReLU(true))
--    layer:add(nn.HardTanh(-3,3))
    layer:add(nn.Tanh())
    return layer
  end

  local function deconv(nIn,nOut,k,s,p)
    local layer = nn.Sequential()
    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p)) 
    layer:add(nn.SpatialBatchNormalization(nOut))
--    layer:add(nn.ReLU(true))
--    layer:add(nn.HardTanh(-3,3))
    return layer
  end

-- add some sizes??? 
  local function shortcut()
    return nn.Identity()
  end

  local function combine(layer, shortcut)
    local inner_L = nn.Sequential()
    :add(nn.ConcatTable()
      :add(layer)
      :add(nn.Identity()))
    :add(nn.CAddTable(true))
    return inner_L
  end

  local function final_layer(L1, L2)
    local L = nn.Sequential()     -- Create a network that takes a Tensor as input
    c = nn.ConcatTable()          -- The same Tensor goes through two different Linear
    c:add(L1)                     -- Layers in Parallel
    c:add(L2)
    L:add(c)
    local dim = 2
    L:add(nn.JoinTable(dim))      -- Finally, the tables are joined together and output, along dim
    return L
  end

  local function scaleLayer(nIn, nOut, k, s, p)
    local layer = nn.Sequential()
    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
    layer = require('weight-init')(layer, method)
    layer:add(nn.SpatialBatchNormalization(nOut))
    return layer
  end
----------------------------------------------------------------------
-- layers inside layers - so that I can create the shortcuts (viz. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  model = nn.Sequential()

  local L1_chan = 32
  local L2_chan = 64
  local L3_chan = 128
  local L4_chan = 256

--  (nIn, nOut, k, s, p)
  model:add(conv(channels*2, L1_chan, 7, 1, 2))
  model:add(conv(L1_chan, L2_chan, 5, 2, 2))
  model:add(conv(L2_chan, L3_chan, 5, 2, 2))
  model:add(conv(L3_chan, L2_chan, 5, 1, 2))
  model:add(conv(L2_chan, 2, 1, 1, 0))
  model:add(scaleLayer(2, 2, 1, 1, 0))

  model:cuda()
  BNInit('nn.SpatialBatchNormalization')
-----------------------------------------------------------------------
  return model
end
