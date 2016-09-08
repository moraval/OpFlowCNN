require 'torch'   -- torch
require 'image'   -- for image transforms
require('cutorch')
require('cunn')
--require('cudnn')
local nn = require 'nn'      -- provides all sorts of trainable modules/layers

local function BNInit(name)
  for k,v in pairs(model:findModules(name)) do
    v.weight:fill(1)
    v.bias:zero()
  end
end
--torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------

function create_model(channels, size1, size2)

  local function conv(nIn, nOut,kT, k, s, p, pP)
    local layer = nn.Sequential()
    layer:add(nn.VolumetricConvolution(nIn, nOut, kT, k, k,s,s,s,p,p,p))
--    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,sW,sH,p,p))
--    layer:add(nn.SpatialBatchNormalization(nOut))
    layer:add(nn.ReLU(true))
    poolModule = nn.VolumetricAveragePooling(kT+1,k,k,s,s,s,p,pP,pP)
    layer:add(poolModule)
    return layer
  end


  local function deconv(nIn,nOut,kT,k,s,pT, p)
    local layer = nn.Sequential()
    layer:add(nn.VolumetricFullConvolution(nIn, nOut, kT, k, k, s,s,s,pT,p,p))
--    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,sW,sH,p,p)) 
    layer:add(nn.ReLU(true)) -- not sure if it should be here or not 
--    layer:add(nn.SpatialBatchNormalization(nOut))
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

  function final_layer(L1, L2)
    local L = nn.Sequential()         -- Create a network that takes a Tensor as input
    c = nn.ConcatTable()          -- The same Tensor goes through two different Linear
    c:add(L1)       -- Layers in Parallel
    c:add(L2)
    L:add(c)
    local dim = 2
    L:add(nn.JoinTable(dim))      -- Finally, the tables are joined together and output, along dim
    return L
  end

  local function inner(nIn)
    innerL = nn.Sequential()
    innerL:add(conv(nIn, nIn*2, 2, 3, 1, 1, 0))
    innerL:add(deconv(nIn*2, nIn, 2, 3, 1, 1, 0))
    return innerL
  end

----------------------------------------------------------------------

  start_chan = channels

----------------------------------------------------------------------
-- layers inside layers - so that I can create the shortcuts (viz. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)

  model = nn.Sequential()

  local L1_chan = start_chan * 6
  local L2_chan = L1_chan * 2
  local L3_chan = L2_chan * 2
  local L4_chan = L3_chan * 2
--  L4_chan = start_chan

  local L0 = nn.Sequential()
  L0:add(conv(L2_chan, L3_chan, 2, 3, 1, 1, 0))
  L0:add(conv(L3_chan, L4_chan, 2, 3, 1, 1, 0))
  L0:add(inner(L4_chan))
  L0:add(deconv(L4_chan, L3_chan, 2, 3, 1, 1, 0))
  L0:add(deconv(L3_chan, L2_chan, 2, 3, 1, 1, 0))

  local L0cS0 = combine(
    L0,
    shortcut())

  local L = nn.Sequential()
  L:add(conv(L1_chan, L2_chan, 2, 3, 1, 1, 0))
  L:add(L0cS0)
  L:add(deconv(L2_chan, L1_chan, 2, 3, 1, 1, 0))


  local LcS = combine(
    L,
    shortcut())

  local L1 = nn.Sequential()
  L1:add(conv(start_chan, L1_chan, 2, 3, 1, 1, 0))
  L1:add(LcS)
  L1:add(deconv(L1_chan, start_chan, 2, 3, 1, 1, 0))

  local L1cS1 = combine(
    L1,
    shortcut())

  model:add(L1cS1)
--  model:add(L0)

   model:add(nn.View(start_chan*2, size1, size2))
   
  local nIn = start_chan*2
  local nOut = 1

-- last layer: creating 2 paths and joining them then together
  local layer1 = nn.Sequential()
  layer1:add(nn.SpatialConvolution(nIn,nOut,1,1,1,1))
  layer1:add(nn.Tanh())

  local layer2 = nn.Sequential()
  layer2:add(nn.SpatialConvolution(nIn,nOut,1,1,1,1))
  layer2:add(nn.Tanh())

  model:add(final_layer(layer1, layer2))
  model:cuda()
  BNInit('nn.SpatialBatchNormalization')
-----------------------------------------------------------------------
  return model
end
