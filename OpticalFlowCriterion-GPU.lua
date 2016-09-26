local OpFlowCriterionGPU, parent = torch.class('nn.OpticalFlowCriterionGPU', 'nn.Criterion')

require 'cutorch'
require 'cudnn'
require 'image'
require 'sys'
require 'os'

local batchSize = 1
local alfa = 0.1
local normalize = 0
local channels, size1, size2 = 3, 16, 16

function OpFlowCriterionGPU:__init(a, norm)
  parent.__init(self)
  alfa = a
  normalize = norm
end
--

--
--input = optical flow (values for u,v), target = image
--function OpFlowCriterion:updateOutput (batchSize, flows, images)
function OpFlowCriterionGPU:updateOutput (flows, images)
  self.output = 0

  batchSize = images:size(1)
  channels = images:size(2)/2
  size1 = images[1][1]:size(1)
  size2 = images[1][1]:size(2)

  image_estimate = torch.CudaTensor(batchSize, 3, size1, size2)

  a = torch.CudaTensor():resizeAs(flows):fill(0)
  b = torch.CudaTensor():resizeAs(flows):fill(0)
  total_variation = torch.CudaTensor():resizeAs(flows):fill(0)
  total_variation_der = torch.CudaTensor():resizeAs(flows):fill(0)

  for i=1,batchSize do
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    local flow = torch.Tensor(2,size1,size2):copy(flows[i])

    image_estimate[i] = image.warp(target, flow, 'bilinear'):cuda()

--    total variation 
    a[i][1]:sub(1,size1-1,1,size2):copy(flows[i][1]:sub(2,size1,1,size2))
    a[i][2]:sub(1,size1,1,size2-1):copy(flows[i][1]:sub(1,size1,2,size2))
    b[i][1]:sub(1,size1-1,1,size2):copy(flows[i][2]:sub(2,size1,1,size2))
    b[i][2]:sub(1,size1,1,size2-1):copy(flows[i][2]:sub(1,size1,2,size2))
 

    tvhelp = torch.CudaTensor():resizeAs(flows[i])
    tvhelp[1]:copy(a[i][1]):mul(-1)
    tvhelp[2]:copy(b[i][1]):mul(-1)
    tvhelp[1] = tvhelp[1]:csub(a[i][2])
    tvhelp[2] = tvhelp[2]:csub(b[i][2])

    flowhelp = torch.CudaTensor():resizeAs(flows[i]):copy(flows[i])

    total_variation_der[i] = tvhelp:add(flowhelp:mul(2))

    flowhelp:copy(flows[i])

    total_variation[i][1]:csub(a[i][1],flowhelp[1]):pow(2):mul(0.5)
    total_variation[i][2]:csub(b[i][1],flowhelp[2]):pow(2):mul(0.5)
    total_variation[i][1]:add(((a[i][2]:csub(flowhelp[1])):pow(2)):mul(0.5))
    total_variation[i][2]:add(((b[i][2]:csub(flowhelp[2])):pow(2)):mul(0.5))
    total_variation[i][1]:pow(0.5)
    total_variation[i][2]:pow(0.5)

    tvhelp:copy(total_variation[i])

    total_variation_der[i] = total_variation_der[i]:cdiv(tvhelp:cmax(0.0000001))
  end

  differences = torch.CudaTensor(batchSize,size1,size2)
  differences = (images:sub(1,batchSize,1,channels) - image_estimate)
  differences = differences:sum(2)
  differences = differences:div(3)

  diffs = torch.Tensor(batchSize,size1,size2):copy(differences)
  local diff_sum = diffs:abs():sum()
  tv = torch.Tensor(batchSize,2,size1,size2):copy(total_variation)
  local tv_sum = tv:sum()

  self.output = (1-alfa) * diff_sum + alfa * tv_sum

-- print(self.output/batchSize..', '..diff_sum/batchSize..', '..tv_sum/batchSize)
  return self.output/batchSize, diff_sum/batchSize, tv_sum/batchSize
end

--


-- calculating error according to all outputs
-- inputs - optical flow (values for u,v), target = image
function OpFlowCriterionGPU:updateGradInput (flows, images)

  self.gradInput = torch.CudaTensor()
  self.gradInput:resizeAs(flows):zero()
  -- self.gradInput:resizeAs(flows)

  -- print('grads before averaging')
  for i=1,batchSize do

    local flow = torch.Tensor(2,size1,size2):copy(flows[i])
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    local orig = images[i]:sub(1,3)

    -- gradients in u,v direction
    local flow_shift = torch.Tensor(2,size1,size2)
    local h = 0.5
    flow_shift:copy(flow)
    flow_shift[1] = flow_shift[1] + h
    local plus_U = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda()
    -- local plus_U = torch.CudaTensor(3,size1,size2):fill(-1)
    -- plus_U:sub(1,3,1,size1-1):copy(orig:sub(1,3,2,size1))

    flow_shift[1] = flow_shift[1] - 2*h
    local minus_U = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda()
    -- local minus_U = torch.CudaTensor(3,size1,size2):fill(-1)
    -- minus_U:sub(1,3,2,size1):copy(orig:sub(1,3,1,size1-1))

    -- local gradU = torch.CudaTensor(3,size1,size2)
    local gradU = torch.CudaTensor(3,size1,size2)
    gradU = gradU:csub(minus_U,plus_U):sum(1):div(3*2*h)
    gradU[plus_U[1]:eq(-1)] = 0
    gradU[plus_U[2]:eq(-1)] = 0
    gradU[minus_U[1]:eq(-1)] = 0
    gradU[minus_U[2]:eq(-1)] = 0

    flow_shift[1]:copy(flow[1])
    flow_shift[2] = flow_shift[2] + h
    local plus_V = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda()
    -- local plus_V = torch.CudaTensor(3,size1,size2):fill(-1)
    -- plus_V:sub(1,3,1,size1,1,size2-1):copy(orig:sub(1,3,1,size1,2,size1))

    flow_shift[2] = flow_shift[2] - 2*h
    local minus_V = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda()
    -- local minus_V = torch.CudaTensor(3,size1,size2):fill(-1)
    -- minus_V:sub(1,3,1,size1,2,size2):copy(orig:sub(1,3,1,size1,1,size2-1))

    local gradV = torch.CudaTensor(3,size1,size2)
    gradV = gradV:csub(minus_V,plus_V):sum(1):div(3*2*h)
    gradV[plus_V[1]:eq(-1)] = 0
    gradV[plus_V[2]:eq(-1)] = 0
    gradV[minus_V[1]:eq(-1)] = 0
    gradV[minus_V[2]:eq(-1)] = 0

    -- print('before')
    -- print(self.gradInput[i])
    self.gradInput[i][1]:cmul(differences[i], gradU):mul(1-alfa)
    self.gradInput[i][2]:cmul(differences[i], gradV):mul(1-alfa)
    -- print('intensity grads')
    -- print(self.gradInput[i])

    gradU = torch.CudaTensor(size1,size2)
    gradV = torch.CudaTensor(size1,size2)

    -- derivatives for total variation
    -- flow U
    local TV_minus_U = torch.CudaTensor(size1,size2):fill(-1)
    local TV_plus_U = torch.CudaTensor(size1,size2):fill(-1)
    TV_minus_U:sub(1,size1-1):copy(total_variation[i][1]:sub(2,size1))
    TV_plus_U:sub(2,size1):copy(total_variation[i][1]:sub(1,size1-1))

    -- gradU:div(gradU:csub(TV_minus_U, TV_plus_U),2)
    gradU = gradU:csub(TV_plus_U, TV_minus_U)
    gradU = gradU:div(2)
    gradU[TV_minus_U:eq(-1)] = 0
    gradU[TV_plus_U:eq(-1)] = 0

    -- flow V
    local TV_minus_V = torch.CudaTensor(size1,size2):fill(-1)
    local TV_plus_V = torch.CudaTensor(size1,size2):fill(-1)
    TV_minus_V:sub(1,size1,1,size2-1):copy(total_variation[i][2]:sub(1,size1,2,size2))
    TV_plus_V:sub(1,size1,2,size2):copy(total_variation[i][2]:sub(1,size1,1,size2-1))

    -- gradV:div(gradV:csub(TV_minus_V, TV_plus_V),2)
    gradV = gradV:csub(TV_plus_V, TV_minus_V)
    gradV = gradV:div(2)
    gradV[TV_minus_V:eq(-1)] = 0
    gradV[TV_plus_V:eq(-1)] = 0

    -- print(gradU)
    -- print(gradV)
    -- print(total_variation_der[i])
    -- total_variation_der[i][1]:cmul(total_variation_der[i][1], gradU)
    -- total_variation_der[i][2]:cmul(total_variation_der[i][2], gradV)
    total_variation_der[i][1]:cmul(total_variation[i][1], gradU)
    total_variation_der[i][2]:cmul(total_variation[i][2], gradV)

    -- print('tv grads')
    -- print(self.total_variation_der[i])
    
    self.gradInput[i]:add(total_variation_der[i]:mul(alfa))

--    clip to [-1, 1]
    self.gradInput[i][self.gradInput[i]:gt(1)] = 1
    self.gradInput[i][self.gradInput[i]:lt(-1)] = -1
    -- print(self.gradInput[i])
  end

  -- print('grads after averaging')
--    batch learning
  gradSums = self.gradInput:sum(1):div(batchSize)
  for i=1,batchSize do
    self.gradInput[i]:copy(gradSums)
  end
  -- print(self.gradInput[1][1])
  -- print(self.gradInput[1][2])

  return self.gradInput
end
--
