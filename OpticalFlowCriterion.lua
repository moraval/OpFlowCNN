local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterion', 'nn.Criterion')
require 'image'
batchSize = 6
--torch.setdefaulttensortype('torch.FloatTensor')

function OpFlowCriterion:__init()
  parent.__init(self)
end


--input = optical flow (values for u,v), target = image
--function OpFlowCriterion:updateOutput (batchSize, flows, images)
function OpFlowCriterion:updateOutput (flows, images)
  self.output = 0

  for i=1,batchSize do
--    warp the optical flow into image
    image_estimate = image.warp(images[i][1], flows[i], 'simple', false)

    self.output = self.output + torch.sum(torch.abs(image_estimate - images[i][2]))
    image_estimate = nil
  end

--  return self.output, self.total_weight_tensor[1]
  return self.output
end

--
--input = optical flow (values for u,v), target = image
function OpFlowCriterion:updateGradInput (flows, images)

  self.gradInput = torch.Tensor()
  self.gradInput:resizeAs(flows):zero()

  for i=1,batchSize do
    start_image = images[i][1]
    target = images[i][2]

    image_estimate_small = image.warp(start_image, flows[i], 'simple', false)

    size1 = target:size(1)
    size2 = target:size(2)

    image_estimate = torch.Tensor(size1 + 2, size2 + 2)

    help = image_estimate:sub(2, size1+1, 2, size2+1)
    help:copy(image_estimate_small)

    image_estimate_small = nil

    diff = torch.FloatTensor(3, size1, size2):fill(0)

--copy the needed shifts to diffU or diffV and then subtract it
    a = diff:sub(2,2, 1,size1, 1,size2)
    b = diff:sub(1,1, 1,size1, 1,size2)
    c = diff:sub(3,3, 1,size1, 1,size2)

    help = image_estimate:sub(2, size1+1, 2, size2+1)
    a = torch.abs(help - target)

    help = image_estimate:sub(1, size1, 2, size2+1)
    b = torch.abs(help - target)

    help = image_estimate:sub(3, size1+2, 2, size2+1)
    c = torch.abs(help - target)

    _, min_dir = torch.min(diff, 1)

    help_tensor = torch.DoubleTensor(1,size1,size2)
    for j=1,size1 do
      for l=1,size2 do
        help_tensor[1][j][l] = min_dir[1][j][l]
      end
    end
    min_dir = nil

    grad = self.gradInput:sub(i,i, 1,1)
    grad:add(help_tensor - 2)
    help_tensor = nil

    diff:fill(0)

    a = diff:sub(2,2, 1,size1, 1,size2)
    b = diff:sub(1,1, 1,size1, 1,size2)
    c = diff:sub(3,3, 1,size1, 1,size2)

    help = image_estimate:sub(2, size1+1, 2, size2+1)
    a = torch.abs(help - target)

    help = image_estimate:sub(2, size1+1, 1, size2)
    b = torch.abs(help - target)

    help = image_estimate:sub(2, size1+1, 3, size2+2)
    c = torch.abs(help - target)

--    grad2 = grad2 + torch.min(diff, 3)[2] - 2
    _, min_dir = torch.min(diff, 1)

    help_tensor = torch.DoubleTensor(1,size1,size2)
    for j=1,size1 do
      for l=1,size2 do
        help_tensor[1][j][l] = min_dir[1][j][l]
      end
    end
    min_dir = nil

    grad2 = self.gradInput:sub(i,i, 2,2)
    grad2:add(help_tensor - 2)
    
    help_tensor = nil
    start_image = nil
    target = nil
    print('Done: '..i)
  end
  
  print('grad input size')
  print(self.gradInput:size())
  return self.gradInput
end





