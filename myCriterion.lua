local OpFlowCriterion, parent = torch.class('nn.OpticalFlowCriterion', 'nn.Criterion')

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

function OpFlowCriterion:__len()
   if (self.weights) then
      return #self.weights
   else
      return 0
   end
end

function OpFlowCriterion:updateOutput (input, target)
  
end

function OpFlowCriterion:updateGradInput (input, target)
  
  end