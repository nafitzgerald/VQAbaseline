require 'paths'
require 'cunn'
require 'nn'

local stringx = require 'pl.stringx'
local file = require 'pl.file'

paths.dofile('opensource_base.lua')
paths.dofile('LinearNB.lua')

local debugger = require 'fb.debugger'
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
        local module_vdata = nn.Sequential():add(nn.SelectTable(1))
        local replicatedMask = nn.Sequential():add(nn.SelectTable(2)):add(nn.Replicate(opt.embed_word,3))
        local lookup = nn.Sequential():add(nn.SelectTable(3)):add(nn.LookupTable(manager_vocab.nvocab_question, opt.embed_word))
        local cat2 = nn.ConcatTable():add(lookup):add(replicatedMask)
        local module_tdata = nn.Sequential():add(cat2):add(nn.CMulTable()):add(nn.Sum(2)) --:add(nn.Tanh())

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
    cmd:option('--savepath', 'model')
    cmd:option('--savetag', 'BOWIMG')
    cmd:option('--inputrep', 'question')
    cmd:option('--recoverfrom', '')
    cmd:option('--runmode', 'train')
    cmd:option('--trainval', 1)
    cmd:option('--trainall', 0)
    cmd:option('--inputmodel', '')
    cmd:option('--resultdir', 'result')
    cmd:option('--datapath', 'data/')

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
    cmd:option('--embed_word', 1024,'the word embedding dimension in baseline')
    cmd:option('--test_during_train', 0, 'whether to compute train error')

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
    local opt = initial_params()
    local method = 'BOWIMG'
    local step_trainval = opt.trainval == 1 --  step for train and valiaton
    local step_trainall = opt.trainall == 1 --  step for combining train2014 and val2014
    opt.method = method
    if opt.savetag == nil or opt.savetag == '' then
        opt.savetag = method .. '_' .. opt.inputrep
    end
    opt.save = paths.concat(opt.savepath, opt.savetag ..'.t7')
    local best_model_save_path = paths.concat(opt.savepath, opt.savetag .. '_BEST.t7')

    local stat = {}
    -- load data inside
    if step_trainval then 
        local state_train, manager_vocab = load_visualqadataset(opt, 'trainval2014_train', nil)
        local state_val, _ = load_visualqadataset(opt, 'trainval2014_val', manager_vocab)

        --state_train.dataset = make_batches(opt, state_train, manager_vocab, 'train')
        --state_val.dataset = make_batches(opt, state_val, manager_vocab, 'val')

	local model, criterion, paramx, paramdx
        if opt.recoverfrom == nil or opt.recoverfrom == '' then
            model, criterion = build_model(opt, manager_vocab)
            paramx, paramdx = model:getParameters()
        else
            print("Recovering from: " .. opt.recoverfrom)
            local f_model = torch.load(opt.recoverfrom)
            manager_vocab = f_model.manager_vocab 
            model, criterion = build_model(opt, manager_vocab)
            paramx, paramdx = model:getParameters()
            paramx:copy(f_model.paramx)
        end
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
    local best_score = 0
        print('start training ...')
        for i = 1, opt.epochs do
            print(method .. ' epoch '..i)
            train_epoch(opt, state_train, manager_vocab, context, 'train')
            _, _, perfs = train_epoch(opt, state_val, manager_vocab, context, 'val')
            -- Accumulate statistics
            stat[i] = {acc, perfs.most_freq, perfs.openend_overall, perfs.multiple_overall}
            -- Adjust the learning rate 
            adjust_learning_rate(i, opt, config_layers)
            local score = perfs.openend_overall
            if score > best_score then
                print('Saving best model (epoch ' .. i .. ') - Acc: ' .. score)
                save_model(opt, manager_vocab, context, best_model_save_path)
                best_score = score
            end
        end
    end
    
    if step_trainall then
        local nEpoch_best = 1
        local acc_openend_best = 0
        if step_trainval then

            -- Select the best train epoch number and combine train2014 and val2014
            for i = 1, #stat do
                if stat[i][3]> acc_openend_best then
                    nEpoch_best = i
                    acc_openend_best = stat[i][3]
                end
            end
          
            print('best epoch number is ' .. nEpoch_best)
            print('best acc is ' .. acc_openend_best)
        else
            nEpoch_best = 100
        end
        -- Combine train2014 and val2014
        local nEpoch_trainAll = nEpoch_best
        local state_train, manager_vocab = load_visualqadataset(opt, 'trainval2014', nil)
        --state_train.dataset = make_batches(opt, state_train, manager_vocab, 'train')
        -- recreate the model  
        local model, criterion = build_model(opt, manager_vocab)
        local paramx, paramdx = model:getParameters()
        local params_current, gparams_current = model:parameters()

        local config_layers, grad_last = config_layer_params(opt, params_current, gparams_current, 1)

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
        
        print('start training on all data ...')
        stat = {}
        for i=1, nEpoch_trainAll do
            print('epoch '..i .. '/' ..nEpoch_trainAll)
            _, _, perfs = train_epoch(opt, state_train, manager_vocab, context, 'train')
            stat[i] = {acc, perfs.most_freq, perfs.openend_overall, perfs.multiple_overall}
            adjust_learning_rate(i, opt, config_layers)

            local modelname_curr = opt.save 
            save_model(opt, manager_vocab, context, modelname_curr)
        end
    end
end

function runTest()
    --load the pre-trained model then evaluate on the test set then generate the csv file that could be submitted to the evaluation server
    local method = 'BOWIMG'
    --local model_path = 'model/BOWIMG.t7'
    local opt = initial_params()
    if opt.savetag == nil or opt.savetag == '' then
        opt.savetag = method .. '_' .. opt.inputrep
    end
    local model_path = nil
    if opt.inputmodel == nil or opt.inputmodel == '' then
        model_path = paths.concat(opt.savepath, opt.savetag .. '_BEST.t7')
    else
        model_path = opt.inputmodel
    end
    local testSet = nil
    local updateIDX = 'test'
    if opt.runmode == 'finaltest' then
        testSet = 'test2015'
    elseif opt.runmode == 'testval' then
        updateIDX = 'val'
        testSet = 'trainval2014_val'
    elseif opt.runmode == 'testtrain' then
        updateIDX = 'val' 
        testSet = 'trainval2014_train'
    else
        testSet = 'test-dev2015' --'test2015' and 'test-dev2015'
    end
    opt.method = method

    -- load pre-trained model 
    local f_model = torch.load(model_path)
    local manager_vocab = f_model.manager_vocab 
    local model, criterion = build_model(opt, manager_vocab)
    local paramx, paramdx = model:getParameters()
    paramx:copy(f_model.paramx)

    -- load test data
    local state_test, _ = load_visualqadataset(opt, testSet, manager_vocab)
    --state_test.dataset = make_batches(opt, state_test, manager_vocab, 'train')

    local context = {
        model = model,
        criterion = criterion,
    }
    -- predict 
    local pred, pred_multi, perfs = train_epoch(opt, state_test, manager_vocab, context, updateIDX)
    
    -- output to csv file to be submitted to the VQA evaluation server
    local file_json_openend = paths.concat(opt.resultdir, 'vqa_OpenEnded_mscoco_' .. testSet .. '_'.. opt.savetag .. '_results.json')
    local file_json_multiple = paths.concat(opt.resultdir, 'vqa_MultipleChoice_mscoco_' .. testSet .. '_'.. opt.savetag .. '_results.json')
    print('output the OpenEnd prediction to JSON file...'..file_json_openend) 
    local choice = 0   
    outputJSONanswer(state_test, manager_vocab, pred, pred_multi, file_json_openend, choice)
    print('output the MultipleChoice prediction to JSON file...'..file_json_multiple) 
    choice = 1
    outputJSONanswer(state_test, manager_vocab, pred, pred_multi, file_json_multiple, choice)

    collectgarbage()
end

opt = initial_params() 
if opt.runmode == 'train' then
    runTrainVal()
    runTest()
else
    runTest()
end
