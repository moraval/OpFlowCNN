require 'torch'   -- torch
require 'image'   -- for image transforms
local nn = require 'nn'      -- provides all sorts of trainable modules/layers
--require 'cunn'

local function BNInit(name)
  for k,v in pairs(model:findModules(name)) do
    v.weight:fill(1)
    v.bias:zero()
  end
end
--torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------

function create_model()

  local function conv(nIn, nOut, k, s, p, pP)
    local layer = nn.Sequential()
    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
--    layer:add(nn.SpatialBatchNormalization(nOut))
    layer:add(nn.ReLU(true))
    poolModule = nn.SpatialMaxPooling(k,k,s,s,pP,pP)
    layer:add(poolModule)
    return layer
  end

  local function deconv(nIn,nOut,k,s,p)
    local layer = nn.Sequential()
    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p)) 
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
    innerL:add(conv(nIn, nIn*2, 3, 1, 1, 0))
    innerL:add(deconv(nIn*2, nIn, 3, 1, 2))
    return innerL
  end

----------------------------------------------------------------------

  start_chan = 2

----------------------------------------------------------------------
-- layers inside layers - so that I can create the shortcuts (viz. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)

  model = nn.Sequential()

  local L1_chan = start_chan * 6
  local L2_chan = L1_chan * 6
  local L3_chan = L2_chan * 4

  local L = nn.Sequential()
  L:add(conv(L1_chan, L2_chan, 3, 1, 1, 0))
  L:add(inner(L2_chan))
  L:add(deconv(L2_chan, L1_chan, 3, 1, 2))

  local LcS = combine(
    L,
    shortcut())

--model:add(LcS)


  local L1 = nn.Sequential()
  L1:add(conv(start_chan, L1_chan, 3, 1, 1, 0))
--  L1:add(conv(L1_chan, L2_chan, 3, 1, 1, 0))
  L1:add(LcS)
--  L1:add(deconv(L2_chan, L1_chan, 3, 1, 2))
  L1:add(deconv(L1_chan, start_chan, 3, 1, 2))

  local L1cS1 = combine(
    L1,
    shortcut())

  model:add(L1cS1)

  local nIn = start_chan
  local nOut = 1


-- last layer: creating 2 paths and joining them then together
  local layer1 = nn.Sequential()
  layer1:add(nn.SpatialConvolution(nIn,nOut,1,1,1,1))
--  layer1:add(nn.ReLU(true))
  layer1:add(nn.Tanh())

  local layer2 = nn.Sequential()
  layer2:add(nn.SpatialConvolution(nIn,nOut,1,1,1,1))
  --layer2:add(nn.ReLU(true))
  layer2:add(nn.Tanh())

  model:add(final_layer(layer1, layer2))

  BNInit('nn.SpatialBatchNormalization')
-----------------------------------------------------------------------
  return model
end
