local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterion', 'nn.Criterion')

require 'cutorch'
require 'image'
require 'sys'
require 'os'

local batchSize = 1
local alfa = 0.1
local normalize = 0
local channels size1, size2 = 3, 16,16

function OpFlowCriterion:__init(dir, printF, a, norm, gt, BSize)
  parent.__init(self)
  alfa = a
  normalize = norm
end
--

--
--input = optical flow (values for u,v), target = image
--function OpFlowCriterion:updateOutput (batchSize, flows, images)
function OpFlowCriterion:updateOutput (flows, images)
  self.output = 0

  batchSize = images:size(1)
  channels = images:size(2)/2
  size1 = images[1][1]:size(1)
  size2 = images[1][1]:size(2)

  image_estimate = cutorch.Tensor(batchSize, 3, size1, size2)

  a = cutorch.Tensor():resizeAs(flows):fill(0.00001)
  b = cutorch.Tensor():resizeAs(flows):fill(0.00001)
  total_variation = cutorch.Tensor():resizeAs(flows)
  total_variation_der = cutorch.Tensor():resizeAs(flows)

  -- local images_l = torch.Tensor(batchSize, size1, size2)

  for i=1,batchSize do
    -- local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    image_estimate[i] = image.warp(images:sub(i,i,channels + 1,channels + channels), flows[i], 'bilinear'):cuda()

    -- local flow = torch.Tensor():resizeAs(flows[i]):copy(flows[i])
    -- images_l[i]:copy(images:sub(i,i,1, channels):sum(2))

--    total variation 
    a[i]:sub(1,2,1,size1-1,1,size2):copy(flows[i]:sub(1,2,2,size1))
    b[i]:sub(1,2,1,size1,1,size2-1):copy(flows[i]:sub(1,2,1,size1,2,size2))

    a2 = cutorch.pow(a[i],2)
    b2 = cutorch.pow(b[i],2)

    ax2 = cutorch.cmul(a[i],flows[i]):cmul(2)
    bx2 = cutorch.cmul(b[i],flows[i]):cmul(2)

    x22 = cutorch.pow(flows[i],2):cmul(2)

    total_variation_der[i] = cutorch.cdiv(a[i]:cmul(-1):csub(b[i]):add(flows[i]):cmul(2), cutorch.cmax(cutorch.pow(a2:add(b2):csub(ax2):csub(bx2):add(x22), 1/2),0.000001))
    
    total_variation[i][1] = cutorch.pow(cutorch.pow(a[i][1]:csub(flows[i][1]),2):add(cutorch.pow(b[i][1]:csub(flows[i][1]),2),1/2))
    total_variation[i][2] = cutorch.pow(cutorch.pow(a[i][2]:csub(flows[i][2]),2):add(cutorch.pow(b[i][2]:csub(flows[i][2]),2),1/2))
  end

  differences = images:csub(1,batchSize,1, channels):sum(2):csub(image_estimate:sum(2))

  differences = differences:div(3)
--  total_variation = total_variation/2

  local diff_sum = cutorch.abs(differences):sum()
  local tv_sum = total_variation:sum()

  self.output = diff_sum + alfa * tv_sum

  return self.output/batchSize, diff_sum/batchSize, tv_sum/batchSize
end

--


-- calculating error according to all outputs
-- inputs - optical flow (values for u,v), target = image
function OpFlowCriterion:updateGradInput (flows, images)

  self.gradInput = cutorch.Tensor()
  self.gradInput:resizeAs(flows):zero()
  -- self.gradInput:resizeAs(flows)

  for i=1,batchSize do

    local flow = cutorch.Tensor():resizeAs(flows[i]):copy(flows[i])
    local target = cutorch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    local img = cutorch.Tensor(channels, size1, size2):copy(images:sub(i,i,1,channels)):sum(1)

-- gradients in u,v direction
    local flow_shift = torch.Tensor():resizeAs(flow)

    -- local plus_1_U, minus_1_U, plus_1_V, minus_1_V = torch.Tensor()

    flow_shift:copy(flow)
    flow_shift[1] = flow_shift[1] + 0.5
    plus_1_U = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda():sum(1):div(3)

    flow_shift[1] = flow_shift[1] - 1
    minus_1_U = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda():sum(1):div(3)

    local gradU = cutorch.csub(minus_1_U,plus_1_U)
    gradU[plus_1_U:eq(-1) or minus_1_U:eq(-1)] = 0

    flow_shift[1]:copy(flow[1])
    flow_shift[2] = flow_shift[2] + 0.5
    plus_1_V = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda():sum(1):div(3)

    flow_shift[2] = flow_shift[2] - 1
    minus_1_V = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda():sum(1):div(3)

    local gradV = cutorch.csub(minus_1_V,plus_1_V)
    gradV[plus_1_V:eq(-1) or minus_1_V:eq(-1)] = 0

    self.gradInput[i][1] = cutorch.cmul(differences[i], gradU)
    self.gradInput[i][2] = cutorch.cmul(differences[i], gradV)

    -- local TV_minus_U = torch.Tensor():resizeAs(flow[1]):copy(total_variation[i][1])
    local TV_minus_U = cutorch.Tensor():resizeAs(flow[1]):fill(-1)
    TV_minus_U:sub(2,size1):copy(total_variation[i]:sub(1,1,1,size1-1))

    gradU = cutorch.cdiv(cutorch.csub(TV_minus_U, a[i][1]),2)
    a[i][1]:sub(1,size1,1,1):fill(-1)
    a[i][1]:sub(1,1,1,size2):fill(-1)
    gradU[TV_minus_U:eq(-1) or a[i][1]:eq(-1)] = 0

    -- local TV_minus_V = torch.Tensor():resizeAs(flow[1]):copy(total_variation[i][2])
    local TV_minus_V = cutorch.Tensor():resizeAs(flow[1]):fill(-1)
    TV_minus_V:sub(1,size1,2,size2):copy(total_variation[i]:sub(2,2,1,size1,1,size2-1))

    gradV = cutorch.cdiv(cutorch.csub(TV_minus_V, b[i][2]),2)
    b[i][2]:sub(1,size1,1,1):fill(-1)
    b[i][2]:sub(1,1,1,size2):fill(-1)
    gradV[TV_minus_V:eq(-1) or b[i][2]:eq(-1)] = 0

    total_variation_der[i][1] = torch.cmul(total_variation_der[i][1], gradU)
    total_variation_der[i][2] = torch.cmul(total_variation_der[i][2], gradV)

    self.gradInput[i] = self.gradInput[i] + total_variation_der[i]:cmul(alfa)

--    clip to [-1, 1]
    self.gradInput[i][self.gradInput[i]:gt(1)] = 1
    self.gradInput[i][self.gradInput[i]:lt(-1)] = -1

  end

--    batch learning
  gradSums = self.gradInput:sum(1):div(batchSize)
  for i=1,batchSize do
    self.gradInput[i]:copy(gradSums)
  end

  return self.gradInput
end
--
