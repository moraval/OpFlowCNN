local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterion', 'nn.Criterion')

require 'image'
require 'sys'

batchSize = 1
epoch = 1
channels = 3
printFlow = false
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
  printFlow = printF
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

  image_estimate = torch.Tensor(batchSize, size1, size2)

  local images_l = torch.Tensor(batchSize, size1, size2)

  for i=1,batchSize do
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))
    local image_estimate_l = image.warp(target, flows[i], 'bilinear')

    if (i==1) then img_est = image_estimate_l end

    image_estimate[i]:copy(image_estimate_l:sum(1))
    images_l[i]:copy(images:sub(i,i,1, channels):sum(2))
  end

  differences = image_estimate - images_l

  self.output = torch.abs(differences):sum()
  return self.output
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

-- gradients in u,v direction
    local flow_shift = torch.Tensor():resizeAs(flow)

    flow_shift:copy(flow)

    flow_shift[1] = flow_shift[1] + 1
    local plus_1_U = image.warp(target, flow_shift, 'bilinear', true):sum(1)
--    plus_1_U[torch.gt(plus_1_U,size1)] = size1

    flow_shift[1] = flow_shift[1] - 2
    local minus_1_U = image.warp(target, flow_shift, 'bilinear', true):sum(1)
--    minus_1_U[torch.lt(minus_1_U,1)] = 1

    local gradU = (minus_1_U - plus_1_U)/2

    flow_shift[1]:copy(flow[1])
    flow_shift[2] = flow_shift[2] + 1
    local plus_1_V = image.warp(target, flow_shift, 'bilinear', true):sum(1) 
--    plus_1_V[torch.gt(plus_1_V,size2)] = size2

    flow_shift[2] = flow_shift[2] - 2
    local minus_1_V = image.warp(target, flow_shift, 'bilinear', true):sum(1)
--    minus_1_V[torch.lt(minus_1_V,1)] = 1

    local gradV = (minus_1_V - plus_1_V)/2

    self.gradInput[i][1] = torch.cmul(differences[i], gradU[1])
    self.gradInput[i][2] = torch.cmul(differences[i], gradV[1])

-- regularize - Total variation
    local a = torch.Tensor():resizeAs(flow):fill(0)
    a:sub(1,2,1,size1-1,1,size2):copy(flow:sub(1,2,2,size1))

    local b = torch.Tensor():resizeAs(flow):fill(0)
    b:sub(1,2,1,size1,1,size2-1):copy(flow:sub(1,2,1,size1,2,size2))

    a2 = torch.pow(a,2)
    b2 = torch.pow(b,2)

    ax = torch.cmul(a,flow)
    bx = torch.cmul(b,flow)

    x2 = torch.pow(flow,2)

    total_variation_der = torch.cdiv(-a-b+2*flow, torch.cmax(torch.pow(a2+b2-ax-bx+2*x2, 1/2),0.000001))

    total_variation = torch.Tensor():resizeAs(flow)

    local flow_iplus = torch.Tensor():resizeAs(flow):fill(0)
    flow_iplus:sub(1,2,1,size1-1):copy(flow:sub(1,2,2,size1))
    local flow_jplus = torch.Tensor():resizeAs(flow):fill(0)
    flow_jplus:sub(1,2,1,size1,1,size2-1):copy(flow:sub(1,2,1,size1,2,size2))

    total_variation[1] = torch.pow(torch.pow(flow_iplus[1] - flow[1],2) + torch.pow(flow_jplus[1] - flow[1],2),1/2)
    total_variation[2] = torch.pow(torch.pow(flow_iplus[2] - flow[2],2) + torch.pow(flow_jplus[2] - flow[2],2),1/2)

    local TV_minus_U = torch.Tensor():resizeAs(flow[1]):fill(0)
    TV_minus_U:sub(2,size1):copy(total_variation:sub(1,1,1,size1-1))

    gradU = (flow_iplus[1] - TV_minus_U)/2
--    gradU = (TV_minus_U - flow_iplus[1])/2

    local TV_minus_V = torch.Tensor():resizeAs(flow[1]):fill(0)
    TV_minus_V:sub(1,size1,2,size2):copy(total_variation:sub(2,2,1,size1,1,size2-1))

    gradV = (flow_jplus[2] - TV_minus_V)/2
--    gradV = (TV_minus_V - flow_jplus[2])/2

    total_variation[1] = torch.cmul(total_variation_der[1], gradU)
    total_variation[2] = torch.cmul(total_variation_der[2], gradV)

--    print(total_variation:sum())
--    self.gradInput[i] = (1-alfa)*self.gradInput[i] + alfa*total_variation

    self.gradInput[i] = self.gradInput[i] + alfa*total_variation

    if (i==1) then 
      local orig = torch.Tensor(3,size1,size2)
      orig:copy(images:sub(i,i,1, channels))

      save_res(flows[i], img_est, orig, self.gradInput[i]) 
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
function save_res(flow, img, orig, gradient)
  if (epoch % 5 == 0) or epoch == 1 and printFlow then
    local printepoch = epoch
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

--    new_img = torch.floor(img)*normalize
--    image.save('results/'..directory..'/images/'..printA..'/new_img_1_'..printepoch..'.png', new_img)

--    orig_norm = torch.floor(orig)*normalize
--    image.save('results/'..directory..'/images/'..printA..'/orig_1'..printepoch..'.png', orig_norm)

--    diff = 255 - torch.abs(orig_norm - new_img)
--    image.save('results/'..directory..'/images/'..printA..'/new_img_diff_1_'..printepoch..'.png', diff)

    print('grad1')
    print(gradient[1])
    print('flow1')
    print(flow[1])

    print('grad2')
    print(gradient[2])
    print('flow2')
    print(flow[2])

    print('img1')
    print(img[1])
--print('orig1')
--print(orig[1])
    local s1 = 2*16 + 2
    local s2 = 4*16 + 3*2
    bigImg = torch.Tensor(3,s1,s2):fill(4)

    local fl1 = flow[1]
    fl1 = fl1 + math.abs(torch.min(fl1))
    fl1 = fl1 * (1/torch.max(fl1))

    local fl2 = flow[2]
    fl2 = fl2 + math.abs(torch.min(fl2))
    fl2 = fl2 * (1/torch.max(fl2))

--  bigImg[1]:sub(1,16,1,16):copy(fl1*8)
--  bigImg[1]:sub(19,34,1,16):copy(fl2*8)
--  bigImg[2]:sub(1,16,1,16):copy(fl1*8)
--  bigImg[2]:sub(19,34,1,16):copy(fl2*8)
--  bigImg[3]:sub(1,16,1,16):copy(fl1*8)
--  bigImg[3]:sub(19,34,1,16):copy(fl2*8)

    bigImg[1]:sub(1,16,1,16):copy(flow[1])
    bigImg[1]:sub(19,34,1,16):copy(flow[2])
    bigImg[2]:sub(1,16,1,16):copy(flow[1])
    bigImg[2]:sub(19,34,1,16):copy(flow[2])
    bigImg[3]:sub(1,16,1,16):copy(flow[1])
    bigImg[3]:sub(19,34,1,16):copy(flow[2])

    bigImg[1]:sub(1,16,19,34):copy(GT[1][1])
    bigImg[1]:sub(19,34,19,34):copy(GT[1][1])
    bigImg[2]:sub(1,16,19,34):copy(GT[1][1])
    bigImg[2]:sub(19,34,19,34):copy(GT[1][1])
    bigImg[3]:sub(1,16,19,34):copy(GT[1][1])
    bigImg[3]:sub(19,34,19,34):copy(GT[1][1])

--print(gradient)
    local gr1 = gradient[1]
    gr1 = gr1 + math.abs(torch.min(gr1))
    gr1 = gr1 * (1/torch.max(gr1))

    local gr2 = gradient[2]
    gr2 = gr2 + math.abs(torch.min(gr2))
    gr2 = gr2 * (1/torch.max(gr2))

    bigImg[1]:sub(1,16,37,52):copy(gr1*8)
    bigImg[1]:sub(19,34,37,52):copy(gr2*8)

    bigImg[1]:sub(1,16,55,70):copy(img[1]*8)
    bigImg[1]:sub(19,34,55,70):copy(orig[1]*8)

    bigImg = bigImg/8
    printA = 5
    image.save('results/'..directory..'/images/'..printA..'/bigImg_'..printepoch..'.png', bigImg)


  end
end