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

  offsets_h = torch.CudaTensor(2, s1*s2):fill(0)
  for i = 1, s1 do
    for j = 1, s2 do
      offsets_h[1][(i-1) * s1 + j] = i
      offsets_h[2][(i-1) * s1 + j] = j
    end
  end

  offsets_h1 = torch.CudaTensor(2, s1, s2):fill(0)
  for i = 1, s1 do
    for j = 1, s2 do
      offsets_h1[1][i][j] = i
      offsets_h1[2][i][j] = j
    end
  end
end
--

--
--input = optical flow (values for u,v), target = image
--function OpFlowCriterion:updateOutput (batchSize, flows, images)
function OpFlowCriterionGPU:updateOutput (flows, images)
  self.output = 0

  -- warptime = 0
  -- resttime = 0
  -- timer = torch.Timer()

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
    -- timer:reset()
    -- local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    -- local flow = torch.Tensor(2,size1,size2):copy(flows[i])


    -- image_estimate[i] = image.warp(target, flow, 'bilinear', 0, 'pad', -1)
    -- print(image_estimate[i])
    -- image.save('orig_estimate.png', image_estimate[i])

    -- print('Time in torch.warp ' ..timer:time().real ..' sec')
    -- timer:reset()

    image_estimate[i] = my_warp_cuda(images[i]:sub(channels + 1,channels + channels), flows[i])
    -- image.save('cpu_estimate.png', image_estimate[i])

    -- image_estimate[i] = gpu_warp(images[i]:sub(channels + 1,channels + channels), flows[i])
    -- print(image_estimate[i])
    -- image.save('gpu_estimate.png', image_estimate[i])

    -- print('Time in cuda.warp ' ..timer:time().real ..' sec')
    -- warptime = warptime + timer:time().real
    -- timer:reset()

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

  -- resttime = resttime + timer:time().real

-- print(self.output/batchSize..', '..diff_sum/batchSize..', '..tv_sum/batchSize)
  return self.output/batchSize, diff_sum/batchSize, tv_sum/batchSize
end

--


-- calculating error according to all outputs
-- inputs - optical flow (values for u,v), target = image
function OpFlowCriterionGPU:updateGradInput (flows, images)

  -- timer:reset()

  self.gradInput = torch.CudaTensor()
  self.gradInput:resizeAs(flows):zero()

  for i=1,batchSize do

    -- timer:reset()

    -- local flow = torch.Tensor(2,size1,size2):copy(flows[i])
    -- local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    -- local orig = images[i]:sub(1,3)

    -- gradients in u,v direction
    local flow_shift = torch.CudaTensor(2,size1,size2)
    local h = 0.5
    flow_shift:copy(flows[i])
    flow_shift[1] = flow_shift[1]:add(h)

    -- resttime = resttime + timer:time().real
    -- timer:reset()

    local plus_U = my_warp_cuda_pad(images[i]:sub(channels + 1,channels + channels), flow_shift)
    -- image.save('plus_U.png', plus_U)
    -- local plus_U = my_warp_cuda(images[i]:sub(channels + 1,channels + channels), flows[i])
    -- plus_U = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda()
    -- image.save('plus_U_orig.png', plus_U)
    -- local plus_U = torch.CudaTensor(3,size1,size2):fill(-1)
    -- plus_U:sub(1,3,1,size1-1):copy(orig:sub(1,3,2,size1))

    flow_shift[1] = flow_shift[1]:csub(2*h)
    local minus_U = my_warp_cuda_pad(images[i]:sub(channels + 1,channels + channels), flow_shift)
    -- image.save('minus_U.png', minus_U)
    -- minus_U = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda()
    -- image.save('minus_U_orig.png', minus_U)

    -- warptime = warptime + timer:time().real
    -- timer:reset()
    
    -- local minus_U = torch.CudaTensor(3,size1,size2):fill(-1)
    -- minus_U:sub(1,3,2,size1):copy(orig:sub(1,3,1,size1-1))

    local gradU = torch.CudaTensor(3,size1,size2)
    gradU = gradU:csub(minus_U,plus_U):sum(1):div(3*2*h)
    gradU[plus_U[1]:eq(-1)] = 0
    gradU[plus_U[2]:eq(-1)] = 0
    gradU[minus_U[1]:eq(-1)] = 0
    gradU[minus_U[2]:eq(-1)] = 0

    flow_shift[1]:copy(flows[i][1])
    flow_shift[2] = flow_shift[2]:add(h)

    -- resttime = resttime + timer:time().real
    -- timer:reset()

    local plus_V = my_warp_cuda_pad(images[i]:sub(channels + 1,channels + channels), flow_shift)
    -- image.save('plus_V.png', plus_V)
    -- plus_V = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda()
    -- image.save('plus_V_orig.png', plus_V)
    -- local plus_V = torch.CudaTensor(3,size1,size2):fill(-1)
    -- plus_V:sub(1,3,1,size1,1,size2-1):copy(orig:sub(1,3,1,size1,2,size1))

    flow_shift[2] = flow_shift[2]:csub(2*h)
    local minus_V  = my_warp_cuda_pad(images[i]:sub(channels + 1,channels + channels), flow_shift)
    -- image.save('minus_V.png', minus_V)
    -- minus_V = image.warp(target, flow_shift, 'bilinear',0,'pad',-1):cuda()
    -- image.save('minus_V_orig.png', minus_V)

    -- warptime = warptime + timer:time().real
    -- timer:reset()

    -- local minus_V = torch.CudaTensor(3,size1,size2):fill(-1)
    -- minus_V:sub(1,3,1,size1,2,size2):copy(orig:sub(1,3,1,size1,1,size2-1))

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

  -- resttime = resttime + timer:time().real

  -- print('Time elapsed for all warps: ' .. warptime .. ' seconds')
  -- print('Time elapsed for all the rest: ' .. resttime .. ' seconds')

  return self.gradInput
end
--


function my_warp_cuda(target, flow, pad)
  local my_est = torch.CudaTensor(3,size1,size2)
  local my_target = torch.CudaTensor(3,size1*size2):copy(target)

  local flowhelp = torch.CudaTensor():resizeAs(flow):copy(flow)
  local ix_1 = flowhelp[1]:add(offsets_h1[1])
  local iy_1 = flowhelp[2]:add(offsets_h1[2])

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
  return my_est
end

function my_warp_cuda_pad(target, flow)
  local my_est = torch.CudaTensor(3,size1,size2)
  local my_target = torch.CudaTensor(3,size1*size2):copy(target)
  local flowhelp = torch.CudaTensor():resizeAs(flow):copy(flow)
  local ix_1 = flowhelp[1]:add(offsets_h1[1])
  local iy_1 = flowhelp[2]:add(offsets_h1[2])

  ix_1[ix_1:lt(1)] = -1000000
  ix_1[ix_1:gt(size1)] = -1000000
  iy_1[iy_1:lt(1)] = -1000000
  iy_1[iy_1:gt(size2)] = -1000000

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
  se = (se:csub(ix_nw)):cmul(iy_1:csub(iy_nw))

  ix_nw = ix_nw:add(-1):mul(size1)
  ix_ne = ix_ne:add(-1):mul(size1)
  -- iy_ne = iy_ne:cmax(1)
  ix_sw = ix_sw:add(-1):mul(size1)
  -- iy_sw = iy_sw:cmin(size2)
  ix_se = ix_se:add(-1):mul(size1)
  -- iy_se = iy_se:cmin(size2)

  local value = my_target[1]:index(1,ix_nw:add(iy_nw)):cmul(nw)
  :add(my_target[1]:index(1,ix_ne:add(iy_ne)):cmul(ne))
  :add(my_target[1]:index(1,ix_sw:add(iy_sw)):cmul(sw))
  :add(my_target[1]:index(1,ix_se:add(iy_se)):cmul(se))

  value[ix_nw:add(iy_nw):lt(-100000)] = 0
  value[ix_ne:add(iy_ne):lt(-100000)] = 0
  value[ix_sw:add(iy_sw):lt(-100000)] = 0
  value[ix_se:add(iy_se):lt(-100000)] = 0
  my_est[1] = value
  my_est[2] = value
  my_est[3] = value
  return my_est
end

function gpu_warp(target, flow)
  
  -- A1 = torch.CudaTensor(2, s1, s2)
  A1_1 = torch.CudaTensor(s1*s2)
  A1_2 = torch.CudaTensor(s1*s2)
  A2_1 = torch.CudaTensor(s1*s2)
  A2_2 = torch.CudaTensor(s1*s2)
  A3_1 = torch.CudaTensor(s1*s2)
  A3_2 = torch.CudaTensor(s1*s2)
  A4_1 = torch.CudaTensor(s1*s2)
  A4_2 = torch.CudaTensor(s1*s2)
  
  -- A1_values = torch.CudaTensor(3, s1, s2)
  A1_values = torch.CudaTensor(3, s1*s2)
  A2_values = torch.CudaTensor(3, s1*s2)
  A3_values = torch.CudaTensor(3, s1*s2)
  A4_values = torch.CudaTensor(3, s1*s2)

  w1 = torch.CudaTensor(2, s1*s2)
  w2 = torch.CudaTensor(2, s1*s2)
  w3 = torch.CudaTensor(2, s1*s2)
  w4 = torch.CudaTensor(2, s1*s2)
  local est = torch.CudaTensor():resizeAs(target):fill(0)
  local indexes = torch.CudaTensor():resizeAs(flow):fill(0)

  -- print(flow)

  -- flowHelp = torch.CudaTensor(2,size1*size2):copy(-flow)
  flowHelp = torch.CudaTensor(2,size1*size2)
  flowHelp[1]:copy(flow[1])
  flowHelp[2]:copy(flow[2])
  helpTarget = torch.CudaTensor(3,size1*size2):copy(target)
  -- helpTarget = helpTarget:reshape(3,size1*size2)
  -- local offsets
  indexes[1]:add(offsets_h[2],flowHelp[2])
  indexes[2]:add(offsets_h[1],flowHelp[1])

  -- print(indexes)
  print(string.format("%.8f", indexes[1][1][1]))

  -- indexes_h = offsets_h + flow
  -- indexes = indexes:reshape(2,16*16)
  -- print(indexes[1]:view(16,16))
  -- print(indexes[2]:view(16,16))
  
  A1_1:copy(indexes[1]):floor()
  A1_2:copy(indexes[2]):floor()
  w1[1]:copy(indexes[1]):csub(A1_1)
  w1[2]:copy(indexes[2]):csub(A1_2)
  w1[1] = w1[1]:cmul(w1[2])
  local A1 = torch.CudaTensor():resizeAs(A1_1):copy(A1_1)
  local A12 = torch.CudaTensor():resizeAs(A1_1):copy(A1_2):add(-1)
  A1 = A1:add(A12:mul(size1))

  A2_1:copy(indexes[1]):ceil()
  A2_2:copy(indexes[2]):floor()
  w2[1]:copy(A2_1):csub(indexes[1])
  w2[2]:copy(indexes[2]):csub(A2_2)
  w2[1] = w2[1]:cmul(w2[2])
  local A2 = torch.CudaTensor():resizeAs(A2_1):copy(A2_1)
  local A22 = torch.CudaTensor():resizeAs(A2_1):copy(A2_2):add(-1)
  A2 = A2:add(A22:mul(size1)) --:add(-2)

  A3_1:copy(indexes[1]):ceil()
  A3_2:copy(indexes[2]):ceil()
  w3[1]:copy(A3_1):csub(indexes[1])
  w3[2]:copy(A3_2):csub(indexes[2])
  w3[1] = w3[1]:cmul(w3[2])
  local A3 = torch.CudaTensor():resizeAs(A3_1):copy(A3_1)
  local A32 = torch.CudaTensor():resizeAs(A2_1):copy(A3_2):add(-1)
  A3 = A3:add(A32:mul(size1)) --:add(-2)

  A4_1:copy(indexes[1]):floor()
  A4_2:copy(indexes[2]):ceil()
  w4[1]:copy(indexes[1]):csub(A4_1)
  w4[2]:copy(A4_2):csub(indexes[2])
  w4[1] = w4[1]:cmul(w4[2])
  local A4 = torch.CudaTensor():resizeAs(A4_1):copy(A4_1)
  local A42 = torch.CudaTensor():resizeAs(A2_1):copy(A4_2):add(-1)
  A4 = A4:add(A42:mul(size1)) --:add(-2)

  print(w1[1]:view(16,16))
  print(w2[1]:view(16,16))
  print(w3[1]:view(16,16))
  print(w4[1]:view(16,16))
  print((w1[1] + w2[1] + w3[1] + w4[1]):view(16,16))

  A1_values[1] = helpTarget[1]:index(1,A1):cmul(w1[1])
  A1_values[2] = helpTarget[2]:index(1,A1):cmul(w1[1])
  A1_values[3] = helpTarget[3]:index(1,A1):cmul(w1[1])
  -- image.save('A1_values-1.png', A1_values:reshape(3,size1,size2))
  -- A1_values[1][A1_1:lt(1)] = 0
  -- A1_values[2][A1_1:lt(1)] = 0
  -- A1_values[3][A1_1:lt(1)] = 0
  -- A1_values[1][A1_2:lt(1)] = 0
  -- A1_values[2][A1_2:lt(1)] = 0
  -- A1_values[3][A1_2:lt(1)] = 0

  helpTarget:copy(target)
  A2_values[1] = helpTarget[1]:index(1,A2):cmul(w2[1])
  A2_values[2] = helpTarget[2]:index(1,A2):cmul(w2[1])
  A2_values[3] = helpTarget[3]:index(1,A2):cmul(w2[1])
  -- image.save('A2_values-1.png', A2_values:reshape(3,size1,size2))
  -- A2_values[1][A2_2:lt(1)] = 0
  -- A2_values[2][A2_2:lt(1)] = 0
  -- A2_values[3][A2_2:lt(1)] = 0
  -- A2_values[1][A2_1:gt(size1)] = 0
  -- A2_values[2][A2_1:gt(size1)] = 0
  -- A2_values[3][A2_1:gt(size1)] = 0

  helpTarget:copy(target)
  A3_values[1] = helpTarget[1]:index(1,A3):cmul(w3[1])
  A3_values[2] = helpTarget[2]:index(1,A3):cmul(w3[1])
  A3_values[3] = helpTarget[3]:index(1,A3):cmul(w3[1])
  -- image.save('A3_values-1.png', A3_values:reshape(3,size1,size2))
  -- A3_values[1][A3_1:gt(size1)] = 0
  -- A3_values[2][A3_1:gt(size1)] = 0
  -- A3_values[3][A3_1:gt(size1)] = 0
  -- A3_values[1][A3_2:gt(size2)] = 0
  -- A3_values[2][A3_2:gt(size2)] = 0
  -- A3_values[3][A3_2:gt(size2)] = 0

  helpTarget:copy(target)
  A4_values[1] = helpTarget[1]:index(1,A4):cmul(w4[1])
  A4_values[2] = helpTarget[2]:index(1,A4):cmul(w4[1])
  A4_values[3] = helpTarget[3]:index(1,A4):cmul(w4[1])
  -- image.save('A4_values-1.png', A4_values:reshape(3,size1,size2))
  -- A4_values[1][A4_1:lt(1)] = 0
  -- A4_values[2][A4_1:lt(1)] = 0
  -- A4_values[3][A4_1:lt(1)] = 0
  -- A4_values[1][A4_2:gt(size2)] = 0
  -- A4_values[2][A4_2:gt(size2)] = 0
  -- A4_values[3][A4_2:gt(size2)] = 0

  est = est:add(A1_values):add(A2_values):add(A3_values):add(A4_values)
  -- image.save('A1_values.png', A1_values:reshape(3,size1,size2))
  -- image.save('A2_values.png', A2_values:reshape(3,size1,size2))
  -- image.save('A3_values.png', A3_values:reshape(3,size1,size2))
  -- image.save('A4_values.png', A4_values:reshape(3,size1,size2))
  -- image.save('est.png', est)
  return est
end