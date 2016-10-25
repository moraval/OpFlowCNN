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
local chan = 1

function create_model(channels, size1, size2, batchSize)

  chan = channels
  s1 = size1
  s2 = size2

  local function conv(nIn, nOut, k, s, p)
    local layer = nn.Sequential()
    layer:add(cudnn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
    layer = require('weight-init.lua')(layer, method)
    layer:add(nn.LeakyReLU())
    return layer
  end

  local function deconv(nIn,nOut,k,s,p)
    local layer = nn.Sequential()
    layer:add(cudnn.SpatialFullConvolution(nIn,nOut,k,k,s,s,p,p)) 
    layer = require('weight-init.lua')(layer, method)
    layer:add(nn.LeakyReLU())
    return layer
  end  

  local function convScale(nIn, nOut, k, s, p)
    local layer = nn.Sequential()
    layer:add(cudnn.SpatialConvolution(nIn,nOut,k,k,s,s,p,p))
    layer = require('weight-init.lua')(layer, method)
    return layer
  end

  local function combine(layer, shortcut)
    local inner_L = nn.Sequential()
    :add(nn.ConcatTable()
      :add(layer)
      :add(shortcut))
    :add(nn.JoinTable(2))
    return inner_L
  end

  local L1_chan = 32
  local L2_chan = 64
  local L3_chan = 128
  local L4_chan = 256/2
  local L5_chan = 512/2

  local model = nn.Sequential()
  model:add(conv(2*chan,L1_chan,5,1,2))

  local L = nn.Sequential()
  L:add(conv(L1_chan,L1_chan,5,2,2))
  L:add(conv(L1_chan,L2_chan,5,1,2))

  local L1 = nn.Sequential()
  L1:add(conv(L2_chan,L2_chan,3,2,1))
  L1:add(conv(L2_chan,L2_chan,3,1,1))
 
  local L2 = nn.Sequential()
  L2:add(conv(L2_chan,L3_chan,3,2,1))
  L2:add(conv(L3_chan,L3_chan,3,1,1))

  L2:add(deconv(L3_chan,L2_chan,4,2,1))
  
  L1:add(combine(L2,nn.Identity()))
  L1:add(deconv(L2_chan*2,L2_chan,4,2,1))
  
  L:add(combine(L1,nn.Identity()))
  L:add(deconv(L2_chan*2,L1_chan,4,2,1))

  model:add(L)
  model:add(convScale(L1_chan,2,1,1,0))
  
--  put on CUDA
  model:cuda()
-----------------------------------------------------------------------
  return model
end
