local saveProcess = false
function save_results(namedir, flow, img, orig, GT, i, epoch, saveGrad, savingProcess, grad)
    saveProcess = savingProcess
    local s1, s2 = 0, 0
    local size1 = flow:size(2)
    local size2 = flow:size(3)
    if saveGrad then
        s1 = 2*size1 + 2
        s2 = 4*size2 + 3*2
    else
        s1 = 2*size1 + 2
        s2 = 3*size2 + 2*2
    end

    local bigImg = torch.Tensor(3,s1,s2):fill(4)

    -- fl1 = flow[1] + math.abs(torch.min(fl1))
    -- fl2 = flow[2] + math.abs(torch.min(fl2))
    local fl1 = (flow[1] + 1)/2
    local fl2 = (flow[2] + 1)/2
    -- print(s1 .. ', ' .. s2 ..'; ' .. size1.. ','..size2)

    bigImg[1]:sub(1,size1,1,size2):copy(fl1)
    bigImg[1]:sub(size1+3,size1*2+2,1,size2):copy(fl2)
    bigImg[2]:sub(1,size1,1,size1):copy(fl1)
    bigImg[2]:sub(size1+3,size1*2+2,1,size2):copy(fl2)
    bigImg[3]:sub(1,size1,1,size1):copy(fl1)
    bigImg[3]:sub(size1+3,size1*2+2,1,size2):copy(fl2)

    -- local GT1 = GT[1] + math.abs(torch.min(GT[1]))
    -- local GT2 = GT[2] + math.abs(torch.min(GT[2]))
    local GT1 = (GT[1] + 1)/2
    local GT2 = (GT[2] + 1)/2

    bigImg[1]:sub(1,size1,size1+3,size1*2+2):copy(GT1)
    bigImg[1]:sub(size1+3,size1*2+2,size1+3,size1*2+2):copy(GT2)
    bigImg[2]:sub(1,size1,size1+3,size1*2+2):copy(GT1)
    bigImg[2]:sub(size1+3,size1*2+2,size1+3,size1*2+2):copy(GT2)
    bigImg[3]:sub(1,size1,size1+3,size1*2+2):copy(GT1)
    bigImg[3]:sub(size1+3,size1*2+2,size1+3,size1*2+2):copy(GT2)


    -- bigImg:sub(1,3,1,16,37,52):copy(img*8)
    -- bigImg:sub(1,3,19,34,37,52):copy(orig*8)

    bigImg:sub(1,3,1,size1,size1*2+5,size1*3+4):copy(img*8)
    bigImg:sub(1,3,size1+3,size1*2+2,size1*2+5,size1*3+4):copy(orig*8)

    if saveGrad then
        grad[1] = grad[1] + math.abs(torch.min(grad[1]))
        grad[2] = grad[2] + math.abs(torch.min(grad[2]))
        bigImg[1]:sub(1,size1,size1*3+4,size1*4):copy(grad[1])
        bigImg[1]:sub(size1+3,size1*2+2,size1*3+4,size1*4 + 3*2):copy(grad[2])
        bigImg[2]:sub(1,size1,size1*3+4,size1*4 + 3*2):copy(grad[1])
        bigImg[2]:sub(size1+3,size1*2+2,size1*3+4,size1*4 + 3*2):copy(grad[2])
        bigImg[3]:sub(1,size1,size1*3+4,size1*4 + 3*2):copy(grad[1])
        bigImg[3]:sub(size1+3,size1*2+2,size1*3+4,size1*4 + 3*2):copy(grad[2])
    end

    bigImg = bigImg/8
    local printEpoch = string.format("%05d", epoch)
    local printI = string.format("%05d", i)

    if saveProcess then
     image.save(namedir .. '/img_'..printI..'_'..printEpoch..'.png', bigImg)   
    else
     image.save(namedir .. '/img_'..printEpoch..'_'..printI..'.png', bigImg)
    end
  end