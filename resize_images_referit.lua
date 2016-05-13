require('loadcaffe')
require('image')
require('lfs')
require('paths')
require('cunn')
require('torch')
local stringx = require 'pl.stringx'
local debugger = require 'fb.debugger'


local resize_to = 512

function do_dir(indir, outdir)
    print ("DOING " .. indir)
    local i = 0
    for file in lfs.dir(indir) do
        i = i + 1
        print(i)
        if stringx.endswith(file, '.jpg') then
            print(file)
            local filename = paths.concat(indir, file)
            local im = nill
            if pcall(function () im = image.load(filename) end) then
                local size = im:size()

                if im:size()[1] == 1 then
                    im = torch.repeatTensor(im, 3, 1, 1)
                end

                scaled = image.scale(im, resize_to)
                ssize = scaled:size()

                padded = torch.zeros(3, resize_to, resize_to)
                if ssize[2] < resize_to then
                    start = 1 + (resize_to - ssize[2])/2
                    padded:narrow(2,start,ssize[2]):copy(scaled)
                else
                    start = 1 + (resize_to - ssize[3])/2
                    padded:narrow(3, start, ssize[3]):copy(scaled)
                end

                name = stringx.split(file, '.')

                path = paths.concat(outdir, name[1] .. '_cropped.jpg')
                image.save(path, padded)
            end
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


--for i = 0, 40 do
for i = 30, 40 do
    local indir = string.format("/home/nfitz/hdd2/data/ReferitData/Images/benchmark/saiapr_tc-12/%02d/images", i)
    local outdir = string.format("/home/nfitz/hdd2/data/ReferitData/images_cropped/")
    do_dir(indir, outdir)
end
os.exit()

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
