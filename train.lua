-- training the model

require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

-- configurations
cutorch.setDevice(1)
--torch.setnumthreads(4)
--torch.setdefaulttensortype('torch.FloatTensor')
local model_dir = "model"
setupLogger(paths.concat(model_dir, 'log.txt'))
paths.dofile(paths.concat(model_dir, 'model.lua'))
gConfig = getConfig()
gConfig.model_dir = model_dir

-- `createModel` is defined in config.lua, it returns the network model and the criterion (loss function)
local model, criterion = createModel(gConfig)
logging(string.format('Model configuration:\n%s', model))
local modelSize, nParamsEachLayer = modelSize(model)
logging(string.format('Model size: %d\n%s', modelSize, nParamsEachLayer))