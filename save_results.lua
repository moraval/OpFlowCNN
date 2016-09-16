function save_results(namedir, flow, img, orig, GT, i, epoch)
    local s1 = 2*16 + 2
    local s2 = 3*16 + 2*2

    local bigImg = torch.Tensor(1,s1,s2):fill(4)

    local fl1 = flow[1]
    local fl2 = flow[2]

    fl1 = fl1 + math.abs(torch.min(fl1))
    fl2 = fl2 + math.abs(torch.min(fl2))

    bigImg[1]:sub(1,16,1,16):copy(fl1)
    bigImg[1]:sub(19,34,1,16):copy(fl2)

    local GT1 = GT[1] + math.abs(torch.min(GT[1]))
    local GT2 = GT[2] + math.abs(torch.min(GT[2]))

    bigImg[1]:sub(1,16,19,34):copy(GT1)
    bigImg[1]:sub(19,34,19,34):copy(GT2)

    bigImg[1]:sub(1,16,37,52):copy(img[1]*8)
    bigImg[1]:sub(19,34,37,52):copy(orig[1]*8)

    bigImg = bigImg/8
    local printEpoch = string.format("%05d", epoch)
    local printI = string.format("%03d", i)

    image.save(namedir .. '/val_img_'..printEpoch..'_'..printI..'.png', bigImg)
  end