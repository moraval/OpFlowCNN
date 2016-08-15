local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterion', 'nn.Criterion')
require 'image'
batchSize = 24
epoch = 5

alfa = 0.2

norm1 = 1
norm2 = 1

function OpFlowCriterion:__init()
  parent.__init(self)
end


--input = optical flow (values for u,v), target = image
--function OpFlowCriterion:updateOutput (batchSize, flows, images)
function OpFlowCriterion:updateOutput (flows, images)
  self.output = 0

  size1 = images[1][1]:size(1)
  size2 = images[1][1]:size(2)

  image_estimate = torch.Tensor(batchSize, 1, size1, size2)
  local targets = torch.Tensor(batchSize, 1, size1, size2)

  for i=1,batchSize do
----    warp the optical flow into image
----    image_estimate = image.warp(images[i][1], flows[i])
    image_estimate[i] = image.warp(images[i][1], flows[i], 'bilinear', true)
    targets[i] = images[i][2]
  end
  
  differences = image_estimate - targets
  
  targets = nil
  
  self.output = torch.sum(torch.abs(image_estimate - targets))
  return self.output
end

--


function regularize(flow, r, s)
  my_sum = torch.Tensor(2)

  local help = flow:sub(1,1,r-1,r+1,s-1,s+1)
  my_sum[1] = help:sum() - help[1][2][2]

  help = flow:sub(2,2,r-1,r+1,s-1,s+1)
  my_sum[2] = help:sum() - help[1][2][2]

  return my_sum/8
end
--
--input = optical flow (values for u,v), target = image
function OpFlowCriterion:updateGradInput (flows, images)

  self.gradInput = torch.Tensor()
  self.gradInput:resizeAs(flows):zero()

  for i=1,batchSize do
    local start_image = images[i][1]
    local target = images[i][2]

    size1 = target:size(1)
    size2 = target:size(2)

    if (epoch % 5 == 0 or epoch == 1) then
      if (i==1) then
        my_string = ''
        for r=1,size1,30 do
          for s=1,size2,60 do
            my_string = my_string..flows[i][1][r][s]..' '
          end
        end
        print('before normalization')
        print(my_string)
      end
    end

    -- NORMALIZE
    flows[i][1] = flows[i][1] *norm1
    flows[i][2] = flows[i][2] *norm2

    if (epoch % 5 == 0 or epoch == 1) then
      if (i==1) then    
        my_string = ''
        my_string_2 = ''
        for r=1,size1,30 do
          for s=1,size2,60 do
            my_string = my_string..flows[i][1][r][s]..' '
          end
        end
        print('after normalization')
        print(my_string)
      end
    end

--    local image_estimate_small = image.warp(start_image, flows[i], 'bilinear', true)
--    image_estimate_small = image.warp(start_image, flows)

    if (epoch % 5 == 0 or epoch == 1) then
      if (i==1) then
        image.save('new_img'..i..'_'..epoch..'.png', image_estimate)
      end
    end

    local flow_enlarged = torch.Tensor(2, size1 + 2, size2 + 2):fill(0)
    flow_enlarged:sub(1,1, 2, size1+1, 2, size2+1):copy(flows[i][1])
    flow_enlarged:sub(2,2, 2, size1+1, 2, size2+1):copy(flows[i][2])
    flow_enlarged = nil

    for r=2,size1+1 do
      for s=2,size2+1 do
      
        of_r, of_s = torch.round(flow_enlarged[1][r][s] + r), torch.round(flow_enlarged[2][r][s] + s)
        reg = regularize(flow_enlarged, r, s)
        if (of_r > 0 and of_r <= size1) and (of_s > 0 and of_s <= size2) then 
        
--          derivatives in the direction of U and V
          dirU = torch.abs(image_estimate[i][1][math.min(of_r+1,size1)][of_s] - image_estimate[i][1][math.max(of_r-1,1)][of_s])
          dirV = torch.abs(image_estimate[i][1][of_r][math.min(of_s+1,size2)] - image_estimate[i][1][of_r][math.max(of_s-1,1)])

          self.gradInput[i][1][r-1][s-1] = differences[i][1][of_r][of_s] * dirU
          self.gradInput[i][2][r-1][s-1] = differences[i][1][of_r][of_s] * dirV
        end
        self.gradInput[i][1][r-1][s-1] = self.gradInput[i][1][r-1][s-1] + alfa*reg[1]
        self.gradInput[i][2][r-1][s-1] = self.gradInput[i][2][r-1][s-1] + alfa*reg[2]
      end
    end
  end

  gradSums = self.gradInput:sum(1)/batchSize
  for i=1,batchSize do
    self.gradInput[i]:copy(gradSums)
  end
  epoch = epoch + 5
  return self.gradInput
end





