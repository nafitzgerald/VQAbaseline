require('loadcaffe')
require('image')
require('lfs')
require('paths')
local stringx = require 'pl.stringx'


local resize_to = 224


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

            d = {}
            d.filename = file
            d.name = name[1]
            d.image = scaled


            path = paths.concat(outdir, name[1] .. '.t7')
            torch.save(path, d)
        end
    end
end

function cache_vgg(indir, outdir)

    model = loadcaffe.load('visModels/VGG_ILSVRC_16_layers_deploy.prototxt', 'visModels/VGG_ILSVRC_16_layers.caffemodel')
    model:remove()
    model:remove()
    model:remove()

    convLayers = nn.Sequential()
    for i = 1,31 do
        convLayers:add(model:get(i))
    end

    topLayers = nn.Sequential()
    for i = 32,model:size() do
        topLayers:add(model:get(i))
    end

    for file in lfs.dir(indir) do
        if stringx.endswith(file, '.t7') then
            -- change this after this run
            local im = torch.load(file)
            name = stringx.split(file, '.')[1]
            outfile = paths.join(indir, name .. '.jpg')
            image.save(outfile, im)
            -- change to here

            convOutput = convLayers:forward(im)
            topOutput = topLayers:forward(convOutput)
            print(convOutpur:size())
            print(topOutput:size())
            os.exit()
        end
    end
end


--local indir = "/home/nfitz/data/VQA/images/test2015/test2015"
--local outdir = "/home/nfitz/data/VQA/images/test2015/test2015_vgg_preproc/"
--do_dir(indir, outdir)

--local indir = "/home/nfitz/data/VQA/images/train2014/train2014"
--local outdir = "/home/nfitz/data/VQA/images/train2014/train2014_vgg_preproc/"
--do_dir(indir, outdir)

--local indir = "/home/nfitz/data/VQA/images/val2014/val2014"
--local outdir = "/home/nfitz/data/VQA/images/val2014/val2014_vgg_preproc/"
--do_dir(indir, outdir)

local indir = "/home/nfitz/data/VQA/images/val2014/val2014_vgg_preproc/"
cache_vgg(indir, outdir)
