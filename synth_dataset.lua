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

function load_dataset(datasize)
  local matio = require 'matio'

  local resdir = 'data/synt_mat/'
--  local name = 'train-1024-dataset-16-pix.mat'
--  local name = 'train-32-dataset-16-pix-cool-bg.mat'
--  local name = 'train-data-diff-pix_all-dir_diff-bg_just-scaled.mat'
--  local name = 'train-data-8-pix_all-dir_diff-bg_just-scaled.ma
--  local name = 'train-big-data-16-pix_all-dir_diff-bg_just-scaled.mat'
--  local name = 'train-big-data-16-pix_only-move_diff-bg_just-scaled.mat'
--  local name = 'train-big-data-16-pix_only-move_cool-bg.mat'
--  local name = 'train-data-8-pix_all-dir_normed-bg_just-scaled.mat'
--  local name = 'train-data-diff-1-pix_all-dir_normed-bg_just-scaled.mat'

  -- local name = 'dataset-2000-diff-sizes-4-moves.mat'
  -- local name = 'dataset-2000-diff-sizes-moves-4-scale-4.mat'
  -- local name = 'train-1024-dataset-restricted-4-16-pix.mat'
  -- local name = 'train-1024-dataset1-restricted-4-16-pix.mat'
  -- local name = 'train-2048-dataset1-restricted-4-16-pix-sc-4.mat'
  local name = 'train-2048-dataset-same-bg-restricted-4-16-pix-sc-2.mat'

  local dataname = resdir .. name

  -- local datasize = 512
  -- local datasize = 256
  -- local datasize = 4
  -- 'data_big','data_small'
  local help1 = matio.load(dataname, 'data_big'):sub(1,datasize)
  local help2 = matio.load(dataname, 'data_small'):sub(1,datasize)
  local help3 = matio.load(dataname, 'gt'):sub(1,datasize)

  return help1, help2, help3, dataname
  -- if (batchSize == 32) then return help1, help2, help3, dataname end

  -- local randInd = math.random(32)

  -- local dataset = help1:sub(randInd,randInd)
  -- local targets = help2:sub(randInd,randInd)
  -- local GT = help3:sub(randInd,randInd)

  -- for i = 1,batchSize-1 do
  --   randInd = math.random(32)
  --   dataset = torch.cat(dataset, help1:sub(randInd,randInd), 1)
  --   targets = torch.cat(targets, help2:sub(randInd,randInd), 1)
  --   GT = torch.cat(GT, help3:sub(randInd,randInd), 1)
  -- end

  -- return dataset, targets, GT, dataname
end

function load_val_dataset(start, datasize)
  local matio = require 'matio'

  local resdir = 'data/synt_mat/'
  -- local name = 'validate-128-dataset-all-8-16-pix.mat'
  -- local name = 'validate-128-dataset1-other-4-16-pix.mat'
  -- local name = 'validate-128-dataset-16-pix.mat'
  -- local name = 'validate-16-dataset-16-pix-cool-bg.mat'

  -- local name = 'dataset-2000-diff-sizes-4-moves.mat'
  -- local name = 'dataset-2000-diff-sizes-moves-4-scale-4.mat'
  -- local name = 'train-2048-dataset1-restricted-4-16-pix-sc-4.mat'
  local name = 'train-2048-dataset-same-bg-restricted-4-16-pix-sc-2.mat'

  local dataname = resdir .. name

  -- 'data_big','data_small'
  -- local help1 = matio.load(dataname, 'data_big'):sub(1,datasize)
  -- local help2 = matio.load(dataname, 'data_small'):sub(1,datasize)
  -- local help3 = matio.load(dataname, 'gt'):sub(1,datasize)

  -- TEST 1:9
  -- local start = 1901
  -- local start = 901
  -- local start = 1
  start = start + 1
  local help1 = matio.load(dataname, 'data_big'):sub(start,datasize)
  local help2 = matio.load(dataname, 'data_small'):sub(start,datasize)
  local help3 = matio.load(dataname, 'gt'):sub(start,datasize)

  return help1, help2, help3, dataname
end

function load_dataset_test()
  local matio = require 'matio'

  local resdir = 'data/synt_mat/'
--  local dataname = 'test-data-10-pix_one-dir_normed-bg_just-scaled.mat'
  -- local dataname = 'test-data-8-pix_one-dir_normed-bg_just-scaled.mat'
  local dataname = 'test-big-data-16-pix_only-move_diff-bg.mat'

  local datasize = 1
  local help1 = matio.load(resdir .. dataname, 'data_big'):sub(1,datasize)
  local help2 = matio.load(resdir .. dataname, 'data_small'):sub(1,datasize)
  local help3 = matio.load(resdir .. dataname, 'gt'):sub(1,datasize)

  return help1, help2, help3, dataname 
end
