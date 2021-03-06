require('loadcaffe')
require('image')
require('lfs')
require('paths')
require('cunn')
require('torch')
local stringx = require 'pl.stringx'
local debugger = require 'fb.debugger'


local resize_to = 448

function do_dir(indir, outdir)
    print ("DOING " .. indir)
    local i = 0
    for file in lfs.dir(indir) do
        i = i + 1
        print(i)
        if stringx.endswith(file, '.jpg') then
            local filename = paths.concat(indir, file)
            local im = image.load(filename)
            local size = im:size()
            crop_to = math.min(size[2], size[3])
            cropped = image.crop(im, "c", crop_to, crop_to)

            if cropped:size()[1] == 1 then
                cropped = torch.repeatTensor(cropped, 3, 1, 1)
            end

            scaled = torch.Tensor(3, resize_to, resize_to)
            scaled = image.scale(scaled, cropped)

            name = stringx.split(file, '.')

            path = paths.concat(outdir, name[1] .. '_cropped.jpg')
            image.save(path, scaled)
        end
    end
end

function cache_vgg(indir, outdir, mean)
    model = loadcaffe.load('visModels/VGG_ILSVRC_16_layers_deploy.prototxt', 'visModels/VGG_ILSVRC_16_layers.caffemodel')

    model:remove()
    model:remove()
    model:remove()

    convLayers = nn.Sequential()
    for i = 1,31 do
        convLayers:add(model:get(i))
    end

    --topLayers = nn.Sequential()
    --for i = 32,model:size() do
    --    topLayers:add(model:get(i))
    --end

    convLayers:evaluate()
    --topLayers:evaluate()
    convLayers:cuda()
    --topLayers:cuda()

    local i = 1
    for file in lfs.dir(indir) do
        if stringx.endswith(file, '.jpg') then
            print(i)
            i = i+1
            -- change this after this run
            local infile = paths.concat(indir, file)
            local im = image.load(infile, 3, 'double')*255
            name = stringx.replace(file, '.jpg', '')
            -- change to here
            im = image.flip(im, 1)
            im = im:cuda()
            im:csub(mean)

            convOutput = convLayers:forward(im)
            --topOutput = topLayers:forward(convOutput)

            convFile = paths.concat(outdir, name .. '_vggConv.t7')
            conv = {}
            conv.data = convOutput
            torch.save(convFile, conv)

            --topFile = paths.concat(outdir, name .. '_vggTop.t7')
            --top = {}
            --top.data = topOutput
            --torch.save(topFile, top)
        end
    end
end


local indir = "/home/nfitz/hdd2/data/VQA/images/test2015/test2015"
local outdir = "/home/nfitz/hdd2/data/VQA/images/VGG16_448/test2015/cropped/"
--do_dir(indir, outdir)

local indir = "/home/nfitz/hdd2/data/VQA/images/train2014"
local outdir = "/home/nfitz/hdd2/data/VQA/images/VGG16_448/train2014/cropped/"
--do_dir(indir, outdir)

local indir = "/home/nfitz/hdd2/data/VQA/images/val2014/val2014"
local outdir = "/home/nfitz/hdd2/data/VQA/images/VGG16_448/val2014/cropped/"
--do_dir(indir, outdir)

local mean = torch.Tensor{103.939, 116.779, 123.68}
mean:resize(3,1,1)
mean = torch.expand(mean, 3, resize_to, resize_to)
mean = mean:cuda()

for _, tag in pairs({'train2014', 'val2014', 'test2015'}) do
    local outdir = string.format("/home/nfitz/hdd2/data/VQA/images/VGG16_448/%s/", tag)
    local indir = string.format("/home/nfitz/hdd2/data/VQA/images/VGG16_448/%s/cropped/", tag)
    cache_vgg(indir, outdir, mean)
end

--local tag = arg[1]
--local outdir = string.format("/home/nfitz/hdd2/data/VQA/images/VGG16_448/%s/", tag)
--local indir = string.format("/home/nfitz/hdd2/data/VQA/images/VGG16_448/%s/cropped/", tag)
--cache_vgg(indir, outdir, mean)
