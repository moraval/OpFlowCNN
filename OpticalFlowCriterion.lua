local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterion', 'nn.Criterion')
require 'image'
--local image = torch.CudaTensor(2,5)

function OpFlowCriterion:__init(weights, sizeAverage)
  parent.__init(self)
  if sizeAverage ~= nil then
    self.sizeAverage = sizeAverage
  else
    self.sizeAverage = true
  end
  if weights then
    assert(weights:dim() == 1, "weights input should be 1-D Tensor")
    self.weights = weights
  end

  self.output_tensor = torch.zeros(1)
  self.total_weight_tensor = torch.ones(1)
  self.target = torch.zeros(1):long()
end


--
--input = optical flow (values for u,v), target = image
function OpFlowCriterion:updateOutput (flow, target, image)
  if type(target) == 'number' then
    if input:type() ~= 'torch.CudaTensor' then
      self.target = self.target:long()
    end
    self.target[1] = target
  elseif target:type() == 'torch.CudaTensor' then
    self.target = target
  else
    self.target = target:long()
  end

--warp the optical flow into image
  image_estimate = image.warp(image, flow, 'simple', false)
  self.output = torch.sum(image_estimate - target)

  return self.output, self.total_weight_tensor[1]
end

--
--input = optical flow (values for u,v), target = image
function OpFlowCriterion:updateGradInput (input, target)
  if type(target) == 'number' then
    self.target[1] = target
  elseif target:type() == 'torch.CudaTensor' then
    self.target = target
  else
    self.target = target:long()
  end

  self.gradInput:resizeAs(input):zero()

  size1 = input:size(1)
  size2 = input:size(2)
  size3 = input:size(3)
  
  diffU = torch.Tensor(size1, size2, 3):fill(0)

--copy the needed shifts to diffU or diffV and then subtract it
  diffU:sub(1,size1, 1,size2, 2,2) = torch.abs(image_estimate - target)
  diffU:sub(1,size1, 1,size2, 1,1) = image_estimate:sub(1,size1-1)
  diffU:sub(1,size1, 1,size2, 1,1) = torch.abs(diffU:sub(1,size1, 1,size2, 1,1) - target)
  diffU:sub(1,size1, 1,size2, 3,3) = image_estimate:sub(2,size1)
  diffU:sub(1,size1, 1,size2, 3,3) = torch.abs(diffU:sub(1,size1, 1,size2, 3,3) - target)

  _, self.gradInput:sub(1,size1) = torch.min(diffU, 3) - 2

  diffV = torch.Tensor(size1, size3, 3):fill(0)

  diffV:sub(1,size1, 1,size2, 2,2) = torch.abs(image_estimate - target)
  diffV:sub(1,size1, 1,size2, 1,1) = image_estimate:sub(1,size1-1)
  diffV:sub(1,size1, 1,size2, 1,1) = torch.abs(diffV:sub(1,size1, 1,size2, 1,1) - target)
  diffV:sub(1,size1, 1,size2, 3,3) = image_estimate:sub(2,size1)
  diffV:sub(1,size1, 1,size2, 3,3) = torch.abs(diffV:sub(1,size1, 1,size2, 3,3) - target)

  _, self.gradInput:sub(1,size1, 1,size2, 2,2) = torch.min(diffV, 3) - 2

  return self.gradInput
end





