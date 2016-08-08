require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- input dimensions
nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128}
filtsize = 5
poolsize = 2

----------------------------------------------------------------------
local function conv(nIn, nOut, k, s, p)
  local layer = nn.Sequential()

  s:add(Convolution(nIn,nOut,k,k,s,s,p,p))
--  s:add(SBatchNorm(n))
  s:add(ReLU(true))
  s:add(SpatialMaxPooling(kW, kH, dW, dH, padW, padH))
  return layer
end

local function deconv(k,s,p)
  local layer = nn.Sequential()
  local pooling = {k,k,s,s,p,p}
  layer:add(nn.SpatialMaxUnpooling(pooling)
  layer:add(ReLU(true))
  return layer
end

local function inner_2
local output_layer = nn.Sequential()
output_layer:add(L1)
output_layer:add(L2)
return output_layer
end


local function one_after_other(L1, L2, L3)
  local output_layer = nn.Sequential()
  output_layer:add(L1)
  output_layer:add(L2)
  output_layer:add(L3)
  return output_layer
end


local function shortcut(par1, par2, par3)
  local layer = nn.Sequential()
  layer:add(nn.Identity())
  return layer
end

local function concat_layer(layer, shortcut)
  local L1 = nn.Sequential()
  :add(nn.ConcatTable()
    :add(layer)
    :add(shortcut))
  return L1
end

-- layers inside layers - so that I can create the shortcuts (viz. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
model = nn.Sequential()

model:add(concat_layer(
    one_after_other(
      conv(bla, bla, bla),
      inner_2,
      deconv(bla, bla, bla)), 
    shortcut(bla, bla, bla)))


-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 3 : standard 2-layer neural network
--      model:add(nn.View(nstates[2]*filtsize*filtsize))
--      model:add(nn.Dropout(0.5))
--      model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
--      model:add(nn.ReLU())
--      model:add(nn.Linear(nstates[3], noutputs))

----------------------------------------------------------------------
print 'model:'
print(model)

