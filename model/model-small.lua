require 'torch'   -- torch
require 'image'   -- for image transforms
require('cutorch')
require('cunn')

local nn = require 'nn' 

--torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- reset weights - select type
local method = 'flow'

function create_model(channels, size1, size2)

  local function conv(nIn, nOut, k, s, p)
    local layer = nn.Sequential()
    layer:add(nn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
    layer = require('weight-init.lua')(layer, method)
    layer:add(nn.SpatialBatchNormalization(nOut))
--    layer:add(nn.Threshold(-10,-10,true))
    layer:add(nn.HardTanh(-10,10))
    return layer
  end

----------------------------------------------------------------------
-- layers inside layers - so that I can create the shortcuts (viz. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  model = nn.Sequential()

  local L1_chan = 32
  local L2_chan = 64
  local L3_chan = 128
  local L4_chan = 256

  local p = 0.3

  model:add(conv(channels*2, L1_chan, 11, 2, 5))
  model:add(conv(L1_chan, L2_chan, 3, 1, 1))
  model:add(nn.SpatialDropout(p))
  model:add(conv(L2_chan, L3_chan, 7, 2, 3))
  model:add(conv(L3_chan, L2_chan, 3, 1, 1))
  model:add(conv(L2_chan, L1_chan, 11, 1, 5))
  model:add(nn.SpatialDropout(p))
  model:add(conv(L1_chan, 2, 1, 1, 0))

--  reset weights
  model = require('weight-init.lua')(model, method)
--  put on CUDA
  model:cuda()
-----------------------------------------------------------------------
  return model
end
