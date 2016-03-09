require 'loadcaffe'
require 'cunn'

--net = caffe.Net('visModels/VGG_ILSVRC_16_layers_deploy.prototxt', 'visModels/VGG_ILSVRC_16_layers.caffemodel', 'test')
--net = caffe.Net('visModels/bvlc_reference_caffenet_deploy.prototxt', 'visModels/bvlc_reference_caffenet.caffemodel', 'train')

--model = loadcaffe.load('visModels/bvlc_reference_caffenet_deploy.prototxt', 'visModels/bvlc_reference_caffenet.caffemodel', 'test')
model = loadcaffe.load('visModels/VGG_ILSVRC_16_layers_deploy.prototxt', 'visModels/VGG_ILSVRC_16_layers.caffemodel')

model:remove()
model:remove()

model:cuda()

testin = torch.Tensor(100, 3,224,224):cuda()
print(model:forward(testin))

