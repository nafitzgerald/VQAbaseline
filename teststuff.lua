require 'image'
--require 'loadcaffe'
require 'cunn'

--net = caffe.Net('visModels/VGG_ILSVRC_16_layers_deploy.prototxt', 'visModels/VGG_ILSVRC_16_layers.caffemodel', 'test')
--net = caffe.Net('visModels/bvlc_reference_caffenet_deploy.prototxt', 'visModels/bvlc_reference_caffenet.caffemodel', 'train')

--model = loadcaffe.load('visModels/bvlc_reference_caffenet_deploy.prototxt', 'visModels/bvlc_reference_caffenet.caffemodel', 'test')
--model = loadcaffe.load('visModels/bvlc_googlenet_deploy.prototxt', 'visModels/bvlc_googlenet.caffemodel')

--print (model)
--model:remove()
--model:remove()
--model:remove()

--model:cuda()

testin = image.load('/home/nfitz/data/VQA/images/train2014/COCO_train2014_000000000009_cropped.jpg')
testin:cuda()

print(testin[{1,1,1}])
print(testin[{2,1,1}])
print(testin[{3,1,1}])



testin = image.flip(testin, 1)


print(testin[{1,1,1}])
print(testin[{2,1,1}])
print(testin[{3,1,1}])
