require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cunn'

--torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
function create_model()

  local function conv(nIn, nOut, k, s, p, pP)
    local layer = nn.Sequential()
    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
--  s:add(SBatchNorm(n))
    layer:add(nn.ReLU(true))
    layer:add(nn.SpatialMaxPooling(k,k,s,s,pP,pP))
    return layer
  end

  local function deconv(nIn,nOut,k,s,p)
    local layer = nn.Sequential()
    layer:add(nn.SpatialFullConvolution(nIn,nOut,k,k,s,s,p,p)) -- use SpatialFullConvolution instead? - called deconvolution, can lower the number of channels
    layer:add(nn.ReLU(true)) -- not sure if it should be here or not 
    return layer
  end

-- add some sizes??? 
  local function shortcut()
--  local layer = nn.Sequential()
--  layer:add(nn.SpatialAveragePooling(1,1,1,1))
--  layer:add(nn.Identity())
    return nn.Identity()
  end

  local function combine(layer, shortcut)
    local L1 = nn.Sequential()
    :add(nn.ConcatTable()
      :add(layer)
      :add(nn.Identity()))
    :add(nn.CAddTable(true))
    return L1
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

----------------------------------------------------------------------

  local start_chan = 2
  local L1_chan = start_chan * 6

----------------------------------------------------------------------
-- layers inside layers - so that I can create the shortcuts (viz. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  model = nn.Sequential()

  local L = nn.Sequential()
  L:add(conv(start_chan, L1_chan, 3, 1, 0))
  L:add(nn.Sequential()
    :add(conv(L1_chan, L1_chan*2, 3, 1, 0))
    :add(deconv(L1_chan*2, L1_chan, 3, 1, 0)))
  L:add(deconv(L1_chan, start_chan, 3, 1, -2))

--model:add(L)
  model:add(
    combine(
      L,
      shortcut()))

  local nIn = start_chan
  local nOut = 1

-- last layer: creating 2 paths and joining them then together
local layer1 = nn.Sequential()
layer1:add(nn.SpatialConvolution(nIn,nOut,1,1,1,1))
layer1:add(nn.ReLU(true))

local layer2 = nn.Sequential()
layer2:add(nn.SpatialConvolution(nIn,nOut,1,1,1,1))
layer2:add(nn.ReLU(true))

model:add(final_layer(layer1, layer2))

-----------------------------------------------------------------------
  return model
end
