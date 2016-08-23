require 'image'
--torch.setdefaulttensortype('torch.FloatTensor')


local function create_train_dataset(input_dir, train_dir, batchSize, channels, size1, size2, name)

  local function zfill(s,N)
    if s:len() < N then
      repeat
        s = '0' .. s
      until s:len() == N
    end 
    return s
  end

--  train_data = torch.Tensor(batchSize, channels * 2, size1, size2)
  train_data = torch.Tensor(128, 6, 94, 311)
--  labels = torch.Tensor(batchSize, channels, size1, size2)

  for idx = 1,batchSize do
    img_1 = string.format(input_dir .. 'in' .. zfill(idx..'',2) .. "a.png")
    img_2 = string.format(input_dir .. 'in' .. zfill(idx..'',2) .. "b.png")
    im1 = image.load(img_1)
    im2 = image.load(img_2)

    data = torch.cat(im1, im2, 1)
    train_data[idx]:sub(1,3,1,1):copy(im1)
    train_data[idx]:sub(1,3,2,2):copy(im2)
--    labels[idx] = im2

  end
  print(#train_data)
  torch.save(train_dir .. 'train_data'..name..'.t7', train_data, 'ascii')
--  torch.save(train_dir .. 'train_labels_small.t7', labels, 'ascii')
  train_data = nil
end

local function create_test_dataset(input_dir, testBatchSize, opt_flows)
  for idx = 1,testBatchSize do
    img_1 = string.format(input_dir .. 'in' .. "%d" .. "a.png", idx)
    img_2 = string.format(input_dir .. 'in' .. "%d" .. "a.png", idx)
    im1 = image.load(img_1)
    im2 = image.load(img_2)

    data = torch.cat(im1, im2, 1)
    test_data[idx] = {data, opt_flows[idx]}

  end

  torch.save('test_data.t7', test_data, 'ascii')
  test_data = nil
end

----------------------------------------------------------------------

input_dir = arg[1]
train_dir = arg[2]
name = arg[3]
batchSize = arg[4]
channels = arg[5]
size1 = arg[6]
size2 = arg[7]

create_train_dataset(input_dir, train_dir, channels, batchSize, size1, size2, name)

