require 'paths'
require 'cunn'
require 'nn'

local stringx = require 'pl.stringx'
local file = require 'pl.file'

paths.dofile('opensource_base.lua')
paths.dofile('LinearNB.lua')

function build_model(opt, manager_vocab) 
    -- function to build up baseline model
    local model
    if opt.method == 'BOW' then
        model = nn.Sequential()
        local module_tdata = nn.LinearNB(manager_vocab.nvocab_question, opt.embed_word)
        model:add(module_tdata)
        model:add(nn.Linear(opt.embed_word, manager_vocab.nvocab_answer))

    elseif opt.method == 'IMG' then
        model = nn.Sequential()
        model:add(nn.Linear(opt.vdim, manager_vocab.nvocab_answer))
    
    elseif opt.method == 'BOWIMG' then
        model = nn.Sequential()
        local module_tdata = nn.Sequential():add(nn.SelectTable(1)):add(nn.LinearNB(manager_vocab.nvocab_question, opt.embed_word))
        local module_vdata = nn.Sequential():add(nn.SelectTable(2))
        local cat = nn.ConcatTable():add(module_tdata):add(module_vdata)
        model:add(cat):add(nn.JoinTable(2))
        model:add(nn.LinearNB(opt.embed_word + opt.vdim, manager_vocab.nvocab_answer))

    else
        print('no such methods')

    end

    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    criterion.sizeAverage = false
    model:cuda()
    criterion:cuda()

    return model, criterion
end

function initial_params()
    local gpuidx = getFreeGPU()
    print('use GPU IDX=' .. gpuidx)
    cutorch.setDevice(gpuidx)

    local cmd = torch.CmdLine()
    
    -- parameters for general setting
    cmd:option('--savepath', 'models/')

    -- parameters for the visual feature
    cmd:option('--vfeat', 'googlenetFC')
    cmd:option('--vdim', 1024)

    -- parameters for data pre-process
    cmd:option('--thresh_questionword',6, 'threshold for the word freq on question')
    cmd:option('--thresh_answerword', 3, 'threshold for the word freq on the answer')
    cmd:option('--batchsize', 100)
    cmd:option('--seq_length', 50)

    -- parameters for learning
    cmd:option('--uniformLR', 0, 'whether to use uniform learning rate for all the parameters')
    cmd:option('--epochs', 100)
    cmd:option('--nepoch_lr', 100)
    cmd:option('--decay', 1.2)
    cmd:option('--embed_word', 512,'the word embedding dimension in baseline')

    -- parameters for universal learning rate
    cmd:option('--maxgradnorm', 20)
    cmd:option('--maxweightnorm', 2000)

    -- parameters for different learning rates for different layers
    cmd:option('--lr_wordembed', 0.8)
    cmd:option('--lr_other', 0.01)
    cmd:option('--weightClip_wordembed', 1500)
    cmd:option('--weightClip_other', 20)

    return cmd:parse(arg or {})
end

function adjust_learning_rate(epoch_num, opt, config_layers)
    -- Every opt.nepoch_lr iterations, the learning rate is reduced.
    if epoch_num % opt.nepoch_lr == 0 then
        for j = 1, #config_layers.lr_rates do
            config_layers.lr_rates[j] = config_layers.lr_rates[j] / opt.decay
        end
    end
end

function runTrainVal()
    local method = 'BOWIMG'
    local testCombine = false
    local opt = initial_params()
    opt.method = method
    opt.save = opt.savepath .. method ..'.t7'

    -- load data inside
    local state_train, manager_vocab = load_visualqadataset(opt, 'trainval2014_train', nil)
    local state_val, _ = load_visualqadataset(opt, 'trainval2014_val', manager_vocab)
    local model, criterion = build_model(opt, manager_vocab)
    local paramx, paramdx = model:getParameters()
    local params_current, gparams_current = model:parameters()

    local config_layers, grad_last = config_layer_params(opt, params_current, gparams_current, 1)

    -- Save variables into context so that train_epoch could use.
    local context = {
        model = model,
        criterion = criterion,
        paramx = paramx,
        paramdx = paramdx,
        params_current = params_current, 
        gparams_current = gparams_current,
        config_layers = config_layers,
        grad_last = grad_last
    }
    print(params_current)
    print('start training ...')
    local stat = {}
    for i = 1, opt.epochs do
        print(method .. ' epoch '..i)
        train_epoch(opt, state_train, manager_vocab, context, 'train')
        train_epoch(opt, state_val, manager_vocab, context, 'val')

        -- Accumulate statistics
        stat[i] = {acc, acc_match_mostfreq, acc_match_openend, acc_match_multiple}
     
        -- Adjust the learning rate 
        adjust_learning_rate(i, opt, config_layers)
    end

    -- Select the best train epoch number and combine train2014 and val2014
    if testCombine then
        nEpoch_best = 1
        acc_openend_best = 0
        for i = 1, #stat do
            if stat[i][3]> acc_openend_best then
                nEpoch_best = i
                acc_openend_best = stat[i][3]
            end
        end
            
        print('best epoch number is ' .. nEpoch_best)
        print('best acc is ' .. acc_openend_best)

        -- Combine train2014 and val2014
        local nEpoch_trainAll = 100 or nEpoch_best
        local state_train, manager_vocab = load_visualqadataset(opt, 'trainval2014', nil)
        print('start training on all data ...')
        local stat = {}
        for i=1, nEpoch_trainAll do
            print('epoch '..i)
            train_epoch(opt, state_train, manager_vocab, context, 'train')
            stat[i] = {acc, acc_match_mostfreq, acc_match_openend, acc_match_multiple}

            adjust_learning_rate(i, opt, config_layers)

            local modelname_curr = opt.save .. '_bestepoch' .. nEpoch_best ..'_going.t7model'
            save_model(opt, manager_vocab, context, modelname_curr)
        end
        stat[nEpoch_best][1] = acc_openend_best
        local modelname_curr = opt.save .. '_bestepoch' .. nEpoch_best ..'_final.t7model'
        save_model(opt, manager_vocab, context, modelname_curr)
    end
end

runTrainVal()