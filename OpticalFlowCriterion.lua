local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterion', 'nn.Criterion')

require 'image'
require 'sys'

batchSize = 24
epoch = 1
channels = 3
printFlow = false
directory = ''
alfa = 0.1
normalize = 0

function OpFlowCriterion:__init(dir, chan, batchS, printF, a, norm)
  parent.__init(self)
  directory = dir
  channels = chan
--  batchSize = batchS
  printFlow = printF
  alfa = a
  print_A = '2'
  normalize = norm
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
    local image_estimate_l = image.warp(target, flows[i], 'bilinear', true)

    if (i==1) then save_res(flows[1], image_estimate_l) end

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
    local sum_flows = torch.Tensor():resizeAs(flow):fill(0)
--    local target = images:sub(i,i,channels + 1,channels + channels)
    local target = torch.Tensor(channels, size1, size2):copy(images:sub(i,i,channels + 1,channels + channels))

    local flow_shift = torch.Tensor():resizeAs(flow)

    flow_shift:fill(0)
    flow_shift:sub(1,1,1,size1-1):copy(flow:sub(1,1,2,size1))
    flow_shift:sub(2,2,1,size1-1):copy(flow:sub(2,2,2,size1))
    sum_flows = sum_flows + flow_shift

    flow_shift:fill(0)
    flow_shift:sub(1,1,2,size1):copy(flow:sub(1,1,1,size1-1))
    flow_shift:sub(2,2,2,size1):copy(flow:sub(2,2,1,size1-1))
    sum_flows = sum_flows + flow_shift

    flow_shift:fill(0)
    flow_shift:sub(2,2,1,size1,1,size2-1):copy(flow:sub(2,2,1,size1,2,size2))
    flow_shift:sub(1,1,1,size1,1,size2-1):copy(flow:sub(1,1,1,size1,2,size2))
    sum_flows = sum_flows + flow_shift

    flow_shift:fill(0)
    flow_shift:sub(2,2,1,size1,2,size2):copy(flow:sub(2,2,1,size1,1,size2-1))
    flow_shift:sub(1,1,1,size1,2,size2):copy(flow:sub(1,1,1,size1,1,size2-1))
    sum_flows = sum_flows + flow_shift

-- in diagonal directions
    flow_shift:fill(0)
    flow_shift:sub(1,1,2,size1,2,size2):copy(flow:sub(1,1,1,size1-1,1,size2-1))
    flow_shift:sub(2,2,2,size1,2,size2):copy(flow:sub(2,2,1,size1-1,1,size2-1))
    sum_flows = sum_flows + flow_shift

    flow_shift:fill(0)
    flow_shift:sub(1,1,1,size1-1,1,size2-1):copy(flow:sub(1,1,2,size1,2,size2))
    flow_shift:sub(2,2,1,size1-1,1,size2-1):copy(flow:sub(2,2,2,size1,2,size2))
    sum_flows = sum_flows + flow_shift

    flow_shift:fill(0)
    flow_shift:sub(1,1,1,size1-1,2,size2):copy(flow:sub(1,1,2,size1,1,size2-1))
    flow_shift:sub(2,2,1,size1-1,2,size2):copy(flow:sub(2,2,2,size1,1,size2-1))
    sum_flows = sum_flows + flow_shift

    flow_shift:fill(0)
    flow_shift:sub(1,1,2,size1,1,size2-1):copy(flow:sub(1,1,1,size1-1,2,size2))
    flow_shift:sub(2,2,2,size1,1,size2-1):copy(flow:sub(2,2,1,size1-1,2,size2))
    sum_flows = sum_flows + flow_shift

    flow_shift:copy(flow)
    flow_shift[1]:copy(flow_shift[1] + 1)
    local plus_1_U = image.warp(target, flow_shift, 'bilinear', true):sum(1)
    flow_shift[1]:copy(flow_shift[1] - 2)
    local minus_1_U = image.warp(target, flow_shift, 'bilinear', true):sum(1)

--    local orig = image.warp(target, flow, 'bilinear', true)
--    image.save('results/'..directory..'/target'..i..'.png', target*normalize)
--    image.save('results/'..directory..'/orig'..i..'.png', orig*normalize)
--    image.save('results/'..directory..'/plus_1_U'..i..'.png', plus_1_U*normalize)
--    image.save('results/'..directory..'/minus_1_U'..i..'.png', minus_1_U*normalize)

    local gradU = (minus_1_U - plus_1_U)/2

    flow_shift[1]:copy(flow_shift[1] + 1)

    flow_shift[2]:copy(flow_shift[2] + 1)
    local plus_1_V = image.warp(target, flow_shift, 'bilinear', true):sum(1)    
    flow_shift[2]:copy(flow_shift[2] - 2)
    local minus_1_V = image.warp(target, flow_shift, 'bilinear', true):sum(1)

--    image.save('results/'..directory..'/plus_1_V'..i..'.png', plus_1_V*normalize)
--    image.save('results/'..directory..'/minus_1_V'..i..'.png', minus_1_V*normalize)

    local gradV = (minus_1_V - plus_1_V)/2

    -- gradients in u,v direction
    self.gradInput[i][1] = torch.cmul(differences[i], gradU[1])
    self.gradInput[i][2] = torch.cmul(differences[i], gradV[1])

--    print('differences')
--    print(differences:sub(1,1,1,9,1,9))
--    print('plus_1_U')
--    print(plus_1_U:sub(1,1,10,19,10,19))
--    print('minus_1_U')
--    print(minus_1_U:sub(1,1,10,19,10,19))

--    print('plus_1_V')
--    print(plus_1_V:sub(1,1,10,19,10,19))
--    print('minus_1_V')
--    print(minus_1_V:sub(1,1,10,19,10,19))

--    print('gradU')
--    print(gradU:sub(1,1,1,9,1,9))
--    print('gradV')
--    print(gradV:sub(1,1,1,9,1,9))
--    print(differences:size())
--    print(self.gradInput[i][1][5][5])
--    print("END!!!!!")
--    break

    -- regularize
    self.gradInput[i] = (1-alfa)*self.gradInput[i] + alfa*sum_flows
--    print('DONE')
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
function save_res(flow, img)
  if (epoch % 10 == 0) and printFlow then
    printepoch = epoch/4
    local out1 = assert(io.open('results/'..directory..'/flows/'..print_A..'/flow_1_1_'..printepoch..'.csv', "w"))
    local out2 = assert(io.open('results/'..directory..'/flows/'..print_A..'/flow_2_1_'..printepoch..'.csv', "w"))
    splitter = ","
    for k=1,size1 do
      for l=1,size2 do
        out1:write(flow[1][k][l])
        out2:write(flow[2][k][l])
        if j == size2 then
          out1:write("\n")
          out2:write("\n")
        else
          out1:write(splitter)
          out2:write(splitter)
        end
      end
    end
    out1:close()
    out2:close()
    image.save('results/'..directory..'/images/'..print_A..'/new_img_1_'..printepoch..'.png', img*normalize)
  end
end