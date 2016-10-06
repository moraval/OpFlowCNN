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
local epoch = 1

function OpFlowCriterionGPU:__init(a, norm, helpImg)
  parent.__init(self)
  alfa = a
  normalize = norm
  local s1, s2 = helpImg:size(1), helpImg:size(2)

  ix_nw = torch.CudaTensor(s1*s2)
  iy_nw = torch.CudaTensor(s1*s2)
  ix_ne = torch.CudaTensor(s1*s2)
  iy_ne = torch.CudaTensor(s1*s2)
  ix_sw = torch.CudaTensor(s1*s2)
  iy_sw = torch.CudaTensor(s1*s2)
  ix_se = torch.CudaTensor(s1*s2)
  iy_se = torch.CudaTensor(s1*s2)

  -- // get surfaces to each neighbor:
  nw = torch.CudaTensor(s1*s2)
  sw = torch.CudaTensor(s1*s2)
  ne = torch.CudaTensor(s1*s2)
  se = torch.CudaTensor(s1*s2)
  iy_se1 = torch.CudaTensor(s1*s2)
  iy_sw1 = torch.CudaTensor(s1*s2)
  iy_1_1 = torch.CudaTensor(s1*s2)

  offsets = torch.CudaTensor(2, s1, s2):fill(0)
  for i = 1, s1 do
    for j = 1, s2 do
      offsets[1][i][j] = i
      offsets[2][i][j] = j
    end
  end
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
    image_estimate[i] = my_warp_cuda(images[i]:sub(channels + 1,channels + channels), flows[i])

    -- total variation 
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

  return self.output/batchSize, diff_sum/batchSize, tv_sum/batchSize
end

--


-- calculating error according to all outputs
-- inputs - optical flow (values for u,v), target = image
function OpFlowCriterionGPU:updateGradInput (flows, images)

  self.gradInput = torch.CudaTensor()
  self.gradInput:resizeAs(flows):zero()

  for i=1,batchSize do

    -- gradients in u,v direction
    local flow_shift = torch.CudaTensor(2,size1,size2)
    local h = 0.5
    flow_shift:copy(flows[i])
    flow_shift[1] = flow_shift[1]:add(h)
    local plus_U = my_warp_cuda_pad(images[i]:sub(channels + 1,channels + channels), flow_shift, 1)

    flow_shift[1] = flow_shift[1]:csub(2*h)
    local minus_U = my_warp_cuda_pad(images[i]:sub(channels + 1,channels + channels), flow_shift, 2)

    local gradU = torch.CudaTensor(3,size1,size2)
    gradU = gradU:csub(minus_U,plus_U):sum(1):div(3*2*h)
    gradU[plus_U[1]:eq(-1)] = 0
    gradU[plus_U[2]:eq(-1)] = 0
    gradU[minus_U[1]:eq(-1)] = 0
    gradU[minus_U[2]:eq(-1)] = 0

    flow_shift[1]:copy(flows[i][1])
    flow_shift[2] = flow_shift[2]:add(h)
    local plus_V = my_warp_cuda_pad(images[i]:sub(channels + 1,channels + channels), flow_shift, 3)

    flow_shift[2] = flow_shift[2]:csub(2*h)
    local minus_V  = my_warp_cuda_pad(images[i]:sub(channels + 1,channels + channels), flow_shift, 4)

    local gradV = torch.CudaTensor(3,size1,size2)
    gradV = gradV:csub(minus_V,plus_V):sum(1):div(3*2*h)
    gradV[plus_V[1]:eq(-1)] = 0
    gradV[plus_V[2]:eq(-1)] = 0
    gradV[minus_V[1]:eq(-1)] = 0
    gradV[minus_V[2]:eq(-1)] = 0

    self.gradInput[i][1]:cmul(differences[i], gradU):mul(1-alfa)
    self.gradInput[i][2]:cmul(differences[i], gradV):mul(1-alfa)

    gradU = torch.CudaTensor(size1,size2)
    gradV = torch.CudaTensor(size1,size2)

    -- derivatives for total variation
    -- flow U
    local TV_minus_U = torch.CudaTensor(size1,size2):fill(-1)
    local TV_plus_U = torch.CudaTensor(size1,size2):fill(-1)
    TV_minus_U:sub(1,size1-1):copy(total_variation[i][1]:sub(2,size1))
    TV_plus_U:sub(2,size1):copy(total_variation[i][1]:sub(1,size1-1))

    gradU = gradU:csub(TV_plus_U, TV_minus_U)
    gradU = gradU:div(2)
    gradU[TV_minus_U:eq(-1)] = 0
    gradU[TV_plus_U:eq(-1)] = 0

    -- flow V
    local TV_minus_V = torch.CudaTensor(size1,size2):fill(-1)
    local TV_plus_V = torch.CudaTensor(size1,size2):fill(-1)
    TV_minus_V:sub(1,size1,1,size2-1):copy(total_variation[i][2]:sub(1,size1,2,size2))
    TV_plus_V:sub(1,size1,2,size2):copy(total_variation[i][2]:sub(1,size1,1,size2-1))

    gradV = gradV:csub(TV_plus_V, TV_minus_V)
    gradV = gradV:div(2)
    gradV[TV_minus_V:eq(-1)] = 0
    gradV[TV_plus_V:eq(-1)] = 0

    total_variation_der[i][1]:cmul(total_variation_der[i][1], gradU)
    total_variation_der[i][2]:cmul(total_variation_der[i][2], gradV)

    self.gradInput[i]:add(total_variation_der[i]:mul(alfa))

--    clip to [-1, 1]
    self.gradInput[i][self.gradInput[i]:gt(1)] = 1
    self.gradInput[i][self.gradInput[i]:lt(-1)] = -1
  end

--    batch learning
  gradSums = self.gradInput:sum(1):div(batchSize)
  for i=1,batchSize do
    self.gradInput[i]:copy(gradSums)
  end

  epoch = epoch + 1
  return self.gradInput
end
--


function my_warp_cuda(target, flow, pad)
  local my_est = torch.CudaTensor(3,size1,size2)
  local my_target = torch.CudaTensor(3,size1*size2):copy(target)

  local flowhelp = torch.CudaTensor():resizeAs(flow):copy(flow)
  local ix_1 = flowhelp[1]:add(offsets[1])
  local iy_1 = flowhelp[2]:add(offsets[2])

  ix_1 = ix_1:cmax(1)
  ix_1 = ix_1:cmin(size1)
  iy_1 = iy_1:cmax(1)
  iy_1 = iy_1:cmin(size2)

  ix_nw:copy(ix_1):floor()
  iy_nw:copy(iy_1):floor()
  ix_ne:copy(ix_nw):add(1)
  iy_ne:copy(iy_nw)
  ix_sw:copy(ix_nw)
  iy_sw:copy(iy_nw):add(1)
  ix_se:copy(ix_ne)
  iy_se:copy(iy_sw)

  -- get surfaces to each neighbor:
  nw:copy(ix_se)
  sw:copy(ix_ne)
  ne:copy(ix_1)
  se:copy(ix_1)
  iy_se1:copy(iy_se)
  iy_sw1:copy(iy_sw)
  iy_1_1:copy(iy_1)

  nw = (nw:csub(ix_1)):cmul(iy_se1:csub(iy_1))
  ne = (ne:csub(ix_sw)):cmul(iy_sw1:csub(iy_1))
  sw = (sw:csub(ix_1)):cmul(iy_1_1:csub(iy_ne))
  se = (se:csub(ix_nw)):cmul(iy_1:csub(iy_nw))

  ix_nw = ix_nw:add(-1):mul(size1)
  ix_ne = ix_ne:cmin(size1):add(-1):mul(size1)
  iy_ne = iy_ne:cmax(1)
  ix_sw = ix_sw:cmax(1):add(-1):mul(size1)
  iy_sw = iy_sw:cmin(size2)
  ix_se = ix_se:cmin(size1):add(-1):mul(size1)
  iy_se = iy_se:cmin(size2)

  local value = my_target[1]:index(1,ix_nw:add(iy_nw)):cmul(nw)
  :add(my_target[1]:index(1,ix_ne:add(iy_ne)):cmul(ne))
  :add(my_target[1]:index(1,ix_sw:add(iy_sw)):cmul(sw))
  :add(my_target[1]:index(1,ix_se:add(iy_se)):cmul(se))
    
  my_est[1] = value
  my_est[2] = value
  my_est[3] = value
  if (epoch == 20) then
    image.save('my_est.png', my_est)
  end
  return my_est
end

function my_warp_cuda_pad(target, flow, i)
  local my_est = torch.CudaTensor(3,size1,size2)
  local my_target = torch.CudaTensor(3,size1*size2):copy(target)
  local flowhelp = torch.CudaTensor():resizeAs(flow):copy(flow)
  local ix_1 = flowhelp[1]:add(offsets[1])
  local iy_1 = flowhelp[2]:add(offsets[2])

  ix_nw:copy(ix_1):floor()
  iy_nw:copy(iy_1):floor()
  ix_ne:copy(ix_nw):add(1)
  iy_ne:copy(iy_nw)
  ix_sw:copy(ix_nw)
  iy_sw:copy(iy_nw):add(1)
  ix_se:copy(ix_ne)
  iy_se:copy(iy_sw)

  -- // get surfaces to each neighbor:
  nw:copy(ix_se)
  sw:copy(ix_ne)
  ne:copy(ix_1)
  se:copy(ix_1)
  iy_se1:copy(iy_se)
  iy_sw1:copy(iy_sw)
  iy_1_1:copy(iy_1)

  nw = (nw:csub(ix_1)):cmul(iy_se1:csub(iy_1))
  ne = (ne:csub(ix_sw)):cmul(iy_sw1:csub(iy_1))
  sw = (sw:csub(ix_1)):cmul(iy_1_1:csub(iy_ne))
  iy_1_1:copy(iy_1)
  se = (se:csub(ix_nw)):cmul(iy_1_1:csub(iy_nw))

  ix_nw = ix_nw:add(-1):mul(size1)
  ix_ne = ix_ne:cmin(size1):add(-1):mul(size1)
  iy_ne = iy_ne:cmax(1)
  ix_sw = ix_sw:cmax(1):add(-1):mul(size1)
  iy_sw = iy_sw:cmin(size2)
  ix_se = ix_se:cmin(size1):add(-1):mul(size1)
  iy_se = iy_se:cmin(size2)
  
  local value = my_target[1]:index(1,ix_nw:add(iy_nw)):cmul(nw)
  value:add(my_target[1]:index(1,ix_ne:add(iy_ne)):cmul(ne))
  value:add(my_target[1]:index(1,ix_sw:add(iy_sw)):cmul(sw))
  value:add(my_target[1]:index(1,ix_se:add(iy_se)):cmul(se))

  value[ix_1:lt(1)] = -1
  value[ix_1:gt(size1)] = -1
  value[iy_1:lt(1)] = -1
  value[iy_1:gt(size2)] = -1
  my_est[1] = value
  my_est[2] = value
  my_est[3] = value

  return my_est
end