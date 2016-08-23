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

function create_model(channels, size1, size2)

  local function conv(nIn, nOut, k, s, p)
    local layer = nn.Sequential()
    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
    layer:add(nn.SpatialBatchNormalization(nOut))
    layer:add(nn.ReLU(true))
    return layer
  end

  local function deconv(nIn,nOut,k,s,p)
    local layer = nn.Sequential()
    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p)) 
    layer:add(nn.SpatialBatchNormalization(nOut))
    layer:add(nn.ReLU(true))
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
    local L = nn.Sequential()     -- Create a network that takes a Tensor as input
    c = nn.ConcatTable()          -- The same Tensor goes through two different Linear
    c:add(L1)                     -- Layers in Parallel
    c:add(L2)
    L:add(c)
    local dim = 2
    L:add(nn.JoinTable(dim))      -- Finally, the tables are joined together and output, along dim
    return L
  end

----------------------------------------------------------------------
-- layers inside layers - so that I can create the shortcuts (viz. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  model = nn.Sequential()

  local L1_chan = 36
  local L2_chan = 72
  local L3_chan = 144

--  (nIn, nOut, k, s, p)
  model:add(conv(channels*2, L1_chan, 7, 2, 2))
  model:add(conv(L1_chan, L2_chan, 3, 2, 1))
  model:add(conv(L2_chan, L3_chan, 3, 1, 1))
  model:add(conv(L3_chan, 18, 1, 1, 0))

  model:add(nn.View(18*23*78))

  model:add(nn.Linear(18*23*78, 6*23*78)) -- 10 input, 25 hidden units
  model:add(nn.Tanh()) -- some hyperbolic tangent transfer function
  model:add(nn.Linear(6*23*78, 2*23*78)) -- 1 output

  model:add(nn.View(2,23,78))

  model:cuda()
  BNInit('nn.SpatialBatchNormalization')
-----------------------------------------------------------------------
  return model
end
