local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterion', 'nn.Criterion')

require 'image'
require 'sys'
require 'os'

local namedir = ''
local batchSize = 1
local epoch = 1
local printFlow = 1
local alfa = 0.1
local normalize = 0
local channels size1, size2 = 3, 16,16
local printFreq = 100
local finEpochs = 1000
local GT = nil

function OpFlowCriterion:__init(dir, printF, a, norm, gt)
  parent.__init(self)
  namedir = dir
--  epoch = 1
  alfa = a
  normalize = norm
  GT = gt
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

  image_estimate = torch.Tensor(batchSize, 3, size1, size2)

  a = torch.Tensor():resizeAs(flows):fill(0.00001)
  b = torch.Tensor():resizeAs(flows):fill(0.00001)
  total_variation = torch.Tensor():resizeAs(flows)
  total_variation_der = torch.Tensor():resizeAs(flows)

  local images_l = torch.Tensor(batchSize, size1, size2)

  for i=1,batchSize do
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    image_estimate[i] = image.warp(target, flows[i], 'bilinear')

    local flow = torch.Tensor():resizeAs(flows[i]):copy(flows[i])
    images_l[i]:copy(images:sub(i,i,1, channels):sum(2))

--    total variation 
    a[i]:sub(1,2,1,size1-1,1,size2):copy(flow:sub(1,2,2,size1))
    b[i]:sub(1,2,1,size1,1,size2-1):copy(flow:sub(1,2,1,size1,2,size2))

    a2 = torch.pow(a[i],2)
    b2 = torch.pow(b[i],2)

    ax = torch.cmul(a[i],flow)
    bx = torch.cmul(b[i],flow)

    x2 = torch.pow(flow,2)

    total_variation_der[i] = torch.cdiv(-a[i]-b[i]+2*flow, torch.cmax(torch.pow(a2+b2-ax-bx+2*x2, 1/2),0.000001))

    total_variation[i][1] = torch.pow(torch.pow(a[i][1] - flow[1],2) + torch.pow(b[i][1] - flow[1],2),1/2)
    total_variation[i][2] = torch.pow(torch.pow(a[i][2] - flow[2],2) + torch.pow(b[i][2] - flow[2],2),1/2)
  end

  differences = images_l - image_estimate:sum(2)

  local diff_sum = torch.abs(differences):sum()
  local tv_sum = total_variation:sum()

  self.output = diff_sum + alfa * tv_sum

  differences = differences/3
  total_variation = total_variation/2

  return self.output/batchSize, diff_sum/batchSize, tv_sum/batchSize
end

--


-- calculating error according to all outputs
-- inputs - optical flow (values for u,v), target = image
function OpFlowCriterion:updateGradInput (flows, images)

  self.gradInput = torch.Tensor()
  self.gradInput:resizeAs(flows):zero()

  for i=1,batchSize do

    local flow = torch.Tensor():resizeAs(flows[i]):copy(flows[i])
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    local img = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,1,channels)):sum(1)

-- gradients in u,v direction
    local flow_shift = torch.Tensor():resizeAs(flow)

    local plus_1_U, minus_1_U, plus_1_V, minus_1_V = torch.Tensor()

    flow_shift:copy(flow)
    flow_shift[1] = flow_shift[1] + 0.5
    plus_1_U = image.warp(target, flow_shift, 'bilinear'):sum(1)

    flow_shift[1] = flow_shift[1] - 1
    minus_1_U = image.warp(target, flow_shift, 'bilinear'):sum(1)

    local gradU = (minus_1_U - plus_1_U)/1

    flow_shift[1]:copy(flow[1])
    flow_shift[2] = flow_shift[2] + 0.5
    plus_1_V = image.warp(target, flow_shift, 'bilinear'):sum(1) 

    flow_shift[2] = flow_shift[2] - 1
    minus_1_V = image.warp(target, flow_shift, 'bilinear'):sum(1)

    local gradV = (minus_1_V - plus_1_V)/1

    self.gradInput[i][1] = torch.cmul(differences[i], gradU)
    self.gradInput[i][2] = torch.cmul(differences[i], gradV)

    local TV_minus_U = torch.Tensor():resizeAs(flow[1]):copy(total_variation[i][1])
    TV_minus_U:sub(2,size1):copy(total_variation[i]:sub(1,1,1,size1-1))

    gradU = (TV_minus_U - a[i][1])/2 -- definitely correct!!

    local TV_minus_V = torch.Tensor():resizeAs(flow[1]):copy(total_variation[i][2])
    TV_minus_V:sub(1,size1,2,size2):copy(total_variation[i]:sub(2,2,1,size1,1,size2-1))

    gradV = (TV_minus_V - b[i][2])/2 -- definitely correct!!

    total_variation_der[i][1] = torch.cmul(total_variation_der[i][1], gradU)
    total_variation_der[i][2] = torch.cmul(total_variation_der[i][2], gradV)

    self.gradInput[i] = self.gradInput[i] + alfa*total_variation_der[i]

--    clip to [-1, 1]
    self.gradInput[i][self.gradInput[i]:gt(1)] = 1
    self.gradInput[i][self.gradInput[i]:lt(-1)] = -1

--    saving results
    if (epoch == finEpochs) then
      local orig = torch.Tensor(3,size1,size2):copy(images:sub(i,i,1, channels))
      save_res(flows[i], image_estimate[i], orig, self.gradInput[i], i) 
    end
    if (i==1 and (printFlow == 1)) then 
      local orig = torch.Tensor(3,size1,size2):copy(images:sub(i,i,1, channels))
      save_res(flows[i], image_estimate[i], orig, self.gradInput[i], i) 
    end
  end

--    batch learning
  gradSums = self.gradInput:sum(1)/batchSize
  for i=1,batchSize do
    self.gradInput[i]:copy(gradSums)
  end
  epoch = epoch + 1

  return self.gradInput
end
--


--
function save_res(flow, img, orig, gradient, i)
  if (epoch % printFreq == 0) or epoch == 1 then
    local s1 = 2*16 + 2
    local s2 = 4*16 + 3*2

    local bigImg = torch.Tensor(1,s1,s2):fill(4)

    local fl1 = flow[1]
    local fl2 = flow[2]

    fl1 = fl1 + math.abs(torch.min(fl1))
    fl2 = fl2 + math.abs(torch.min(fl2))

    bigImg[1]:sub(1,16,1,16):copy(fl1)
    bigImg[1]:sub(19,34,1,16):copy(fl2)

    local GT1 = GT[i][1] + math.abs(torch.min(GT[1][1]))
    local GT2 = GT[i][2] + math.abs(torch.min(GT[1][2]))

    bigImg[1]:sub(1,16,19,34):copy(GT1)
    bigImg[1]:sub(19,34,19,34):copy(GT2)

    local gr1 = gradient[1]
    gr1 = gr1 + math.abs(torch.min(gr1))
    gr1 = gr1 * (1/torch.max(gr1))

    local gr2 = gradient[2]
    gr2 = gr2 + math.abs(torch.min(gr2))
    gr2 = gr2 * (1/torch.max(gr2))

    bigImg[1]:sub(1,16,37,52):copy(gr1)
    bigImg[1]:sub(19,34,37,52):copy(gr2)

    bigImg[1]:sub(1,16,55,70):copy(img[1]*8)
    bigImg[1]:sub(19,34,55,70):copy(orig[1]*8)

    bigImg = bigImg/8
    local printEpoch = string.format("%05d", epoch)

    if (epoch == finEpochs) then
      image.save(namedir .. '/final/images/bigImg_'..printEpoch..'_' .. i ..'.png', bigImg)
    else
      image.save(namedir .. '/images/bigImg_'..printEpoch..'.png', bigImg)
    end
  end
end