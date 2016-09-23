function save_results(namedir, flow, img, orig, GT, i, epoch, saveGrad, grad)
    local s1, s2 = 0, 0
    if saveGrad then
        s1 = 2*16 + 2
        s2 = 4*16 + 3*2
    else
        s1 = 2*16 + 2
        s2 = 3*16 + 2*2
    end

    local bigImg = torch.Tensor(3,s1,s2):fill(4)

    local fl1 = flow[1]
    local fl2 = flow[2]

    fl1 = fl1 + math.abs(torch.min(fl1))
    fl2 = fl2 + math.abs(torch.min(fl2))

    bigImg[1]:sub(1,16,1,16):copy(fl1)
    bigImg[1]:sub(19,34,1,16):copy(fl2)
    bigImg[2]:sub(1,16,1,16):copy(fl1)
    bigImg[2]:sub(19,34,1,16):copy(fl2)
    bigImg[3]:sub(1,16,1,16):copy(fl1)
    bigImg[3]:sub(19,34,1,16):copy(fl2)

    local GT1 = GT[1] + math.abs(torch.min(GT[1]))
    local GT2 = GT[2] + math.abs(torch.min(GT[2]))

    bigImg[1]:sub(1,16,19,34):copy(GT1)
    bigImg[1]:sub(19,34,19,34):copy(GT2)
    bigImg[2]:sub(1,16,19,34):copy(GT1)
    bigImg[2]:sub(19,34,19,34):copy(GT2)
    bigImg[3]:sub(1,16,19,34):copy(GT1)
    bigImg[3]:sub(19,34,19,34):copy(GT2)

    bigImg:sub(1,3,1,16,37,52):copy(img*8)
    bigImg:sub(1,3,19,34,37,52):copy(orig*8)

    if saveGrad then
        grad[1] = grad[1] + math.abs(torch.min(grad[1]))
        grad[2] = grad[2] + math.abs(torch.min(grad[2]))
        bigImg[1]:sub(1,16,55,70):copy(grad[1])
        bigImg[1]:sub(19,34,55,70):copy(grad[2])
        bigImg[2]:sub(1,16,55,70):copy(grad[1])
        bigImg[2]:sub(19,34,55,70):copy(grad[2])
        bigImg[3]:sub(1,16,55,70):copy(grad[1])
        bigImg[3]:sub(19,34,55,70):copy(grad[2])
    end

    bigImg = bigImg/8
    local printEpoch = string.format("%05d", epoch)
    local printI = string.format("%05d", i)

    image.save(namedir .. '/img_'..printEpoch..'_'..printI..'.png', bigImg)
  end