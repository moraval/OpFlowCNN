local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterionGPU', 'nn.Criterion')

require 'cutorch'
require 'cudnn'
require 'image'
require 'sys'
require 'os'

local batchSize = 1
local alfa = 0.1
local normalize = 0
local channels, size1, size2 = 3, 16, 16

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

  image_estimate = torch.CudaTensor(batchSize, 3, size1, size2)

  a = torch.CudaTensor():resizeAs(flows):fill(0.00001)
  b = torch.CudaTensor():resizeAs(flows):fill(0.00001)
  total_variation = torch.CudaTensor():resizeAs(flows)
  total_variation_der = torch.CudaTensor():resizeAs(flows)

  -- local images_l = torch.Tensor(batchSize, size1, size2)

  for i=1,batchSize do
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    local flow = torch.Tensor(2,size1,size2):copy(flows[i])

    image_estimate[i] = image.warp(target, flow, 'bilinear'):cuda()

    -- images_l[i]:copy(images:sub(i,i,1, channels):sum(2))

--    total variation 
    a[i]:sub(1,2,1,size1-1,1,size2):copy(flows[i]:sub(1,2,2,size1))
    b[i]:sub(1,2,1,size1,1,size2-1):copy(flows[i]:sub(1,2,1,size1,2,size2))

    a2 = torch.CudaTensor():resizeAs(flows[i])
    b2 = torch.CudaTensor():resizeAs(flows[i])
    ax2 = torch.CudaTensor():resizeAs(flows[i])
    bx2 = torch.CudaTensor():resizeAs(flows[i])
    x22 = torch.CudaTensor():resizeAs(flows[i])

    a2:pow(a[i],2)
    b2:pow(b[i],2)
    ax2:mul(ax2:cmul(a[i],flows[i]),2)
    bx2:mul(bx2:cmul(b[i],flows[i]),2)
    x22:mul(x22:pow(flows[i],2),2)

    aihelp = torch.CudaTensor():resizeAs(flows[i]):copy(a[i])
    flowhelp = torch.CudaTensor():resizeAs(flows[i]):copy(flows[i])

    total_variation_der[i] = aihelp:mul(-1):csub(b[i])
    flowhelp:mul(2)

    total_variation_der[i]:add(flowhelp)

    total_variation_der[i]:cdiv((a2:add(b2):csub(ax2):csub(bx2):add(x22)):pow(1/2):cmax(0.000001))

    flowhelp = torch.CudaTensor():resizeAs(flows[i]):copy(flows[i])
    bihelp = torch.CudaTensor():resizeAs(flows[i]):copy(b[i])

    total_variation[i][1]:csub(a[i][1],flowhelp[1]):pow(2)
    total_variation[i][2]:csub(a[i][2],flowhelp[2]):pow(2)

    total_variation[i][1]:add((b[i][1]:csub(flowhelp[1])):pow(2))
    total_variation[i][2]:add((b[i][2]:csub(flowhelp[2])):pow(2))

    total_variation[i][1]:pow(0.5)
    total_variation[i][2]:pow(0.5)

    -- everything the same as in CPU
    -- total_variation[i][1]:add((aihelp[1]:csub(flowhelp[1])):pow(2),(bihelp[1]:csub(flowhelp[1])):pow(2)):pow(0.5)
    -- total_variation[i][2]:add((aihelp[2]:csub(flowhelp[2])):pow(2),(bihelp[2]:csub(flowhelp[2])):pow(2)):pow(0.5)
    -- print(total_variation[i][1][1][1])
    -- print(total_variation[i][2][1][1])
  end

  differences = torch.CudaTensor(batchSize,size1,size2)
  differences = (images:sub(1,batchSize,1, channels):sum(2) - image_estimate:sum(2))
  differences:div(3)
--  total_variation = total_variation/2

  diffs = torch.Tensor(batchSize,size1,size2):copy(differences)
  local diff_sum = diffs:abs():sum()
  tv = torch.Tensor(batchSize,2,size1,size2):copy(total_variation)
  local tv_sum = tv:sum()

  self.output = diff_sum + alfa * tv_sum

-- print(self.output/batchSize..', '..diff_sum/batchSize..', '..tv_sum/batchSize)
  return self.output/batchSize, diff_sum/batchSize, tv_sum/batchSize
end

--


-- calculating error according to all outputs
-- inputs - optical flow (values for u,v), target = image
function OpFlowCriterion:updateGradInput (flows, images)

  self.gradInput = torch.CudaTensor()
  self.gradInput:resizeAs(flows):zero()
  -- self.gradInput:resizeAs(flows)

  -- print('grads before averaging')
  for i=1,batchSize do

    local flow = torch.Tensor(2,size1,size2):copy(flows[i])
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))

-- gradients in u,v direction
    local flow_shift = torch.Tensor(2,size1,size2)

    flow_shift:copy(flow)
    flow_shift[1] = flow_shift[1] + 0.5
    plus_1_U = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda():sum(1):div(3)

    flow_shift[1] = flow_shift[1] - 1
    minus_1_U = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda():sum(1):div(3)

    local gradU = torch.CudaTensor(2,size1,size2)
    gradU:csub(minus_1_U,plus_1_U)
    gradU[plus_1_U:eq(-1)] = 0
    gradU[minus_1_U:eq(-1)] = 0

    flow_shift[1]:copy(flow[1])
    flow_shift[2] = flow_shift[2] + 0.5
    plus_1_V = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda():sum(1):div(3)

    flow_shift[2] = flow_shift[2] - 1
    minus_1_V = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda():sum(1):div(3)

    local gradV = torch.CudaTensor(2,size1,size2)
    gradV:csub(minus_1_V,plus_1_V)
    gradV[plus_1_V:eq(-1)] = 0
    gradV[minus_1_V:eq(-1)] = 0

    self.gradInput[i][1]:cmul(differences[i], gradU)
    self.gradInput[i][2]:cmul(differences[i], gradV)

    -- local TV_minus_U = torch.Tensor():resizeAs(flow[1]):copy(total_variation[i][1])
    local TV_minus_U = torch.CudaTensor(size1,size2):fill(-1)
    TV_minus_U:sub(2,size1):copy(total_variation[i]:sub(1,1,1,size1-1))

    gradU:div(gradU:csub(TV_minus_U, a[i][1]),2)
    a[i][1]:sub(1,size1,1,1):fill(-1)
    a[i][1]:sub(1,1,1,size2):fill(-1)
    gradU[TV_minus_U:eq(-1)] = 0
    gradU[a[i][1]:eq(-1)] = 0

    -- local TV_minus_V = torch.Tensor():resizeAs(flow[1]):copy(total_variation[i][2])
    local TV_minus_V = torch.CudaTensor(size1,size2):fill(-1)
    TV_minus_V:sub(1,size1,2,size2):copy(total_variation[i]:sub(2,2,1,size1,1,size2-1))

    gradV:div(gradV:csub(TV_minus_V, b[i][2]),2)
    b[i][2]:sub(1,size1,1,1):fill(-1)
    b[i][2]:sub(1,1,1,size2):fill(-1)
    gradV[TV_minus_V:eq(-1)] = 0
    gradV[b[i][2]:eq(-1)] = 0

    total_variation_der[i][1]:cmul(total_variation_der[i][1], gradU)
    total_variation_der[i][2]:cmul(total_variation_der[i][2], gradV)

    self.gradInput[i]:add(self.gradInput[i],total_variation_der[i]:mul(alfa))

--    clip to [-1, 1]
    self.gradInput[i][self.gradInput[i]:gt(0.5)] = 0.5
    self.gradInput[i][self.gradInput[i]:lt(-0.5)] = -0.5
    -- print(self.gradInput[i])
  end

  -- print('grads after averaging')
--    batch learning
  gradSums = self.gradInput:sum(1):div(batchSize)
  for i=1,batchSize do
    self.gradInput[i]:copy(gradSums)
  end
  -- print(self.gradInput[i][1])

  return self.gradInput
end
--
