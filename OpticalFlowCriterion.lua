local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterion', 'nn.Criterion')

require 'image'
require 'sys'
require 'os'

der2 = 1
batchSize = 16
epoch = 1
channels = 3
printFlow = 1
directory = ''
alfa = 0.1
normalize = 0
local size1, size2 = 16,16
GT = torch.Tensor(batchSize,2,size1,size2)
img_est = torch.Tensor(3,size1,size2)

function OpFlowCriterion:__init(dir, chan, batchS, printF, a, norm, gt)
  parent.__init(self)
  directory = dir
  channels = chan
--  batchSize = batchS
--  printFlow = printF
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

  size1 = images[1][1]:size(1)
  size2 = images[1][1]:size(2)

--  print('-------------------FLOW---------------------------')
--  print(flows)
--  print('-------------------END---------------------------')
  image_estimate = torch.Tensor(batchSize, size1, size2)

  a = torch.Tensor():resizeAs(flows):fill(0.00001)
  b = torch.Tensor():resizeAs(flows):fill(0.00001)
  total_variation = torch.Tensor():resizeAs(flows)
  total_variation_der = torch.Tensor():resizeAs(flows)

  local images_l = torch.Tensor(batchSize, size1, size2)

  for i=1,batchSize do
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    local image_estimate_l = image.warp(target, flows[i], 'bilinear')
    local flow = torch.Tensor():resizeAs(flows[i]):copy(flows[i])

    if (i==1) then img_est = image_estimate_l end

    image_estimate[i]:copy(image_estimate_l:sum(1))
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

  differences = images_l - image_estimate

  self.output = torch.abs(differences):sum() + alfa * total_variation:sum()
  differences = differences/3
  image_estimate = image_estimate/3
  return self.output/batchSize
end

--



-- calculating error according to all outputs
-- inputs - optical flow (values for u,v), target = image
function OpFlowCriterion:updateGradInput (flows, images)

  if (epoch == 1) then os.execute("mkdir " .. 'results/'..directory..'/images/'..alfa) end
  self.gradInput = torch.Tensor()
  self.gradInput:resizeAs(flows):zero()

  for i=1,batchSize do

    local flow = torch.Tensor():resizeAs(flows[i]):copy(flows[i])
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    local img = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,1,channels)):sum(1)

-- gradients in u,v direction
    local flow_shift = torch.Tensor():resizeAs(flow)

    local plus_1_U, minus_1_U, plus_1_V, minus_1_V = torch.Tensor()

    if der2 == 1 then
      flow_shift:copy(flow)
      flow_shift[1] = flow_shift[1] + 0.5
      plus_1_U = image.warp(target, flow_shift, 'bilinear'):sum(1)

      flow_shift[1] = flow_shift[1] - 1
      minus_1_U = image.warp(target, flow_shift, 'bilinear'):sum(1)
    else
      plus_1_U = torch.Tensor():resizeAs(image_estimate[i]):copy(image_estimate[i])
      plus_1_U:sub(1,size1-1):copy(image_estimate[i]:sub(2,size1))

      minus_1_U = torch.Tensor():resizeAs(image_estimate[i]):copy(image_estimate[i])
      minus_1_U:sub(2,size1):copy(image_estimate[i]:sub(1,size1-1))
    end

    local gradU = (minus_1_U - plus_1_U)/1

    if der2 == 1 then
      flow_shift[1]:copy(flow[1])
      flow_shift[2] = flow_shift[2] + 0.5
      plus_1_V = image.warp(target, flow_shift, 'bilinear'):sum(1) 

      flow_shift[2] = flow_shift[2] - 1
      minus_1_V = image.warp(target, flow_shift, 'bilinear'):sum(1)
    else
      plus_1_V = torch.Tensor():resizeAs(image_estimate[i]):copy(image_estimate[i])
      plus_1_V:sub(1,size1, 1, size2-1):copy(image_estimate[i]:sub(1,size1, 2, size2))

      minus_1_V = torch.Tensor():resizeAs(image_estimate[i]):copy(image_estimate[i])
      minus_1_V:sub(1,size1, 2, size2):copy(image_estimate[i]:sub(1,size1, 1, size2-1))
    end

    local gradV = (minus_1_V - plus_1_V)/1

    self.gradInput[i][1] = torch.cmul(differences[i], gradU)
    self.gradInput[i][2] = torch.cmul(differences[i], gradV)

--    if (printFlow == 1) then 
--      save_grad(gradU, gradV, self.gradInput[i], img_est) 
--    end

    local TV_minus_U = torch.Tensor():resizeAs(flow[1]):copy(total_variation[i][1])
    TV_minus_U:sub(2,size1):copy(total_variation[i]:sub(1,1,1,size1-1))

    gradU = (TV_minus_U - a[i][1])/2

    local TV_minus_V = torch.Tensor():resizeAs(flow[1]):copy(total_variation[i][2])
    TV_minus_V:sub(1,size1,2,size2):copy(total_variation[i]:sub(2,2,1,size1,1,size2-1))

    gradV = (TV_minus_V - b[i][2])/2

    total_variation_der[i][1] = torch.cmul(total_variation_der[i][1], gradU)
    total_variation_der[i][2] = torch.cmul(total_variation_der[i][2], gradV)

    self.gradInput[i] = self.gradInput[i] + alfa*total_variation_der[i]

--    clip to [-1, 1]
    self.gradInput[i][self.gradInput[i]:gt(1)] = 1
    self.gradInput[i][self.gradInput[i]:lt(-1)] = -1

    if (i==1 and (printFlow == 1)) then 
      local orig = torch.Tensor(3,size1,size2)
      orig:copy(images:sub(i,i,1, channels))

      save_res(flows[i], img_est, orig, self.gradInput[i], target) 
    end

  end
  gradSums = self.gradInput:sum(1)/batchSize
  for i=1,batchSize do
    self.gradInput[i]:copy(gradSums)
  end
  epoch = epoch + 1
  return self.gradInput
end
--


--
function save_res(flow, img, orig, gradient, target)
  if (epoch % 10 == 0) or epoch == 1 then
--  local printA = alfa * 10
--  local out1 = assert(io.open('results/'..directory..'/flows/'..printA..'/flow_1_1_'..printepoch..'.csv', "w"))
--  local out2 = assert(io.open('results/'..directory..'/flows/'..printA..'/flow_2_1_'..printepoch..'.csv', "w"))
--  local splitter = ","
--  for k=1,size1 do
--    for l=1,size2 do
--      out1:write(flow[1][k][l])
--      out2:write(flow[2][k][l])
--      if j == size2 then
--        out1:write("\n")
--        out2:write("\n")
--      else
--        out1:write(splitter)
--        out2:write(splitter)
--      end
--    end
--  end
--  out1:close()
--  out2:close()

--    print('max in grads')
--    print(torch.max(gradient[1]) .. ' ' .. torch.max(gradient[2]))
--    print('min in grads')
--    print(torch.min(gradient[1]) .. ' ' .. torch.min(gradient[2]))

    local s1 = 2*16 + 2
    local s2 = 4*16 + 3*2
--    local s1 = 2*8 + 2
--    local s2 = 4*8 + 3*2

    local bigImg = torch.Tensor(1,s1,s2):fill(4)

    local fl1 = flow[1]
    fl1 = fl1 + math.abs(torch.min(fl1))
    fl1 = fl1 * (1/math.max(torch.max(fl1),0.0001))
--    print('flow')
--    print(flow)
--    print('min a max')
--    print(torch.min(flow[1]) .. ' ' .. torch.min(fl1))
--    print(torch.max(flow[1]) .. ' ' .. torch.max(fl1))

    local fl2 = flow[2]
    fl2 = fl2 + math.abs(torch.min(fl2))
    fl2 = fl2 * (1/math.max(torch.max(fl2),0.0001))
--    print('min a max flow2')
--    print(torch.min(flow[2]) .. ' ' .. torch.min(fl2))
--    print(torch.max(flow[2]) .. ' ' .. torch.max(fl2))

--    local flow1 = flow[1] + math.abs(torch.min(flow[1]))
--    local flow2 = flow[2] + math.abs(torch.min(flow[2]))

    bigImg[1]:sub(1,16,1,16):copy(fl1)
    bigImg[1]:sub(19,34,1,16):copy(fl2)

    local GT1 = GT[1][1] + math.abs(torch.min(GT[1][1]))
    local GT2 = GT[1][2] + math.abs(torch.min(GT[1][2]))

    bigImg[1]:sub(1,16,19,34):copy(GT1)
    bigImg[1]:sub(19,34,19,34):copy(GT2)
--    bigImg[1]:sub(1,8,1,8):copy(flow[1])
--    bigImg[1]:sub(11,18,1,8):copy(flow[2])

--    bigImg[1]:sub(1,8,11,18):copy(GT[1][1])
--    bigImg[1]:sub(11,18,11,18):copy(GT[1][1])

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
--    bigImg[1]:sub(1,8,21,28):copy(gr1)
--    bigImg[1]:sub(11,18,21,28):copy(gr2)

--    bigImg[1]:sub(1,8,31,38):copy(img[1]*8)
--    bigImg[1]:sub(11,18,31,38):copy(orig[1]*8)

    bigImg = bigImg/8
    local printA = math.ceil(alfa*10)
    local printEpoch = string.format("%05d", epoch)
    image.save('results/'..directory..'/images/'..alfa..'/bigImg_'..printEpoch..'.png', bigImg)

  end
end
--
function save_grad(gradU, gradV, gradient, img)
  if (epoch % 10 == 0) or epoch == 1 then

    local s1 = 2*16 + 2
    local s2 = 3*16 + 2*2

--    local s1 = 2*8 + 2
--    local s2 = 3*8 + 2*2
    local bigImg = torch.Tensor(1,s1,s2):fill(4)

    diff = differences[1] + math.abs(torch.min(differences[1]))
    diff = diff * (1/torch.max(diff))

--    bigImg[1]:subU(1,8,1,8):copy(diff)
    bigImg[1]:sub(1,16,1,16):copy(diff)

    bigImg[1]:sub(19,34,1,16):copy(img[1])

    gradU = gradU + math.abs(torch.min(gradU))
    gradU = gradU * (1/torch.max(gradU))

    gradV = gradV + math.abs(torch.min(gradV))
    gradV = gradV * (1/torch.max(gradV))
    
    bigImg[1]:sub(1,16,19,34):copy(gradU*8)
    bigImg[1]:sub(19,34,19,34):copy(gradV*8)

    local gr1 = gradient[1]
    gr1 = gr1 + math.abs(torch.min(gr1))
    gr1 = gr1 * (1/torch.max(gr1))

    local gr2 = gradient[2]
    gr2 = gr2 + math.abs(torch.min(gr2))
    gr2 = gr2 * (1/torch.max(gr2))

    bigImg[1]:sub(1,16,37,52):copy(gr1*8)
    bigImg[1]:sub(19,34,37,52):copy(gr2*8)

    bigImg = bigImg/8
    local printA = math.ceil(alfa*10)
    local printEpoch = string.format("%05d", epoch)
    image.save('results/'..directory..'/images/'..alfa..'/gradCheck_'..printEpoch..'.png', bigImg)

  end
end