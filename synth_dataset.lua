require 'image'
--require 'csvigo'

function create_dataset(dir)

  local dataSize = 1
  local sizebig = 64
  local sizesmall = 16
  local size = 12
  local scale = 4

--  local dataset = torch.rand(dataSize,6,sizebig,sizebig)
--  local gauss = image.gaussian(3)
  local dataset = torch.Tensor(dataSize,6,sizebig,sizebig):fill(0)
  local targets = torch.Tensor(dataSize,6,sizesmall,sizesmall):fill(0)
  local GT = torch.Tensor(dataSize,2,sizesmall,sizesmall):fill(0)

  for i = 1,dataSize do
    local c1 = math.random(16,38)
    local cp = 4
    dataset[i]:sub(1,3,c1,c1+size,c1,c1+size):fill(1)

    c1 = c1+cp
    dataset[i]:sub(4,6,c1,c1+size,c1,c1+size):fill(1)

    targets[i]:copy(torch.floor(image.scale(dataset[i]:sub(1,6), sizesmall, sizesmall)))

    GT[i][1][targets[i][1]:eq(1)] = cp/4
    GT[i][2][targets[i][1]:eq(1)] = cp/4
  end

  return dataset, targets, GT
end

function load_dataset(batchSize)
  local matio = require 'matio'

--local dataname = 'data/synt_mat/data-8-pix_all-dir_normed-bg.mat'
  local dataname = 'data/synt_mat/data-10-pix_all-dir_normed-bg_just-scaled.mat'
--local dataname = 'data/synt_mat/data-allnorm-biggerrectless-onedir.mat'
--local dataname = 'data/synt_mat/data-allnorm-biggerrectless-twodir.mat'
--dataname = 'data/synt_mat/data-allnorm-biggerrect-twodir.mat'
--dataname = 'data/synt_mat/data-allnorm-rectless-onedir.mat'
--  'data_big','data_small'
  local help1 = matio.load(dataname, 'data_big')
  local help2 = matio.load(dataname, 'data_small')
  local help3 = matio.load(dataname, 'gt')

  if (batchSize == 32) then return help1, help2, help3, dataname end

  local randInd = math.random(32)

  local dataset = help1:sub(randInd,randInd)
  local targets = help2:sub(randInd,randInd)
  local GT = help3:sub(randInd,randInd)

  for i = 1,batchSize-1 do
    randInd = math.random(32)
    dataset = torch.cat(dataset, help1:sub(randInd,randInd), 1)
    targets = torch.cat(targets, help2:sub(randInd,randInd), 1)
    GT = torch.cat(GT, help3:sub(randInd,randInd), 1)
  end

  return dataset, targets, GT, dataname
end

--load_dataset()