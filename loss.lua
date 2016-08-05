require 'cutorch'
-- calculating energy of the prediction

function calculate_loss (image_estimate, image_gt)

  -- calculate loss for every pixel + add the regularization
  losses = image_estimate - image_gt 
  -- calculate total loss - for comparison
  total_loss = cutorch.sum(losses)
  return total_loss, losses
end

-- is it local or global??
function calculate_variation(optical_flow)

  return variation
end