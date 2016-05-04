require('lfs')
require('image')
local debugger = require 'fb.debugger'
local stringx = require 'pl.stringx'
local file = require 'pl.file'

function getFreeGPU()
    -- select the most available GPU to train
    local nDevice = cutorch.getDeviceCount()
    local memSet = torch.Tensor(nDevice)
    for i=1, nDevice do
        local tmp, _ = cutorch.getMemoryUsage(i)
        memSet[i] = tmp
    end
    local _, curDeviceID = torch.max(memSet,1)
    return curDeviceID[1]
end

-- Here we specify different learning rate and gradClip for different layers. 
-- This is *critical* for the performance of BOW. 
function config_layer_params(opt, params_current, gparams_current, IDX_wordembed)
    local lr_wordembed = opt.lr_wordembed
    local lr_other = opt.lr_other
    local weightClip_wordembed = opt.weightClip_wordembed
    local weightClip_other = opt.weightClip_other

    print("lr_wordembed = " .. lr_wordembed)
    print("lr_other = " .. lr_other)
    print("weightClip_wordembed = " .. weightClip_wordembed)
    print("weightClip_other = " .. weightClip_other)

    local gradientClip_dummy = 0.1
    local weightRegConsts_dummy = 0.000005
    local initialRange_dummy = 0.1
    local moments_dummy = 0.9

    -- Initialize specification of layers.
    local config_layers = {
        lr_rates = {},
        gradientClips = {},
        weightClips = {},
        moments = {},
        weightRegConsts = {},
        initialRange = {}
    }

    -- grad_last is used to add momentum to the gradient.
    local grad_last = {}
    if IDX_wordembed == 1 then
        -- assume wordembed matrix is the params_current[1]
        config_layers.lr_rates = {lr_wordembed}
        config_layers.gradientClips = {gradientClip_dummy}
        config_layers.weightClips = {weightClip_wordembed}
        config_layers.moments = {moments_dummy}
        config_layers.weightRegConsts = {weightRegConsts_dummy}
        config_layers.initialRange = {initialRange_dummy}
        for i = 2, #params_current do
            table.insert(config_layers.lr_rates, lr_other)
            table.insert(config_layers.moments, moments_dummy)
            table.insert(config_layers.gradientClips, gradientClip_dummy)
            table.insert(config_layers.weightClips, weightClip_other)
            table.insert(config_layers.weightRegConsts, weightRegConsts_dummy)
            table.insert(config_layers.initialRange, initialRange_dummy)    
        end

    else
        for i = 1, #params_current do
            table.insert(config_layers.lr_rates, lr_other)
            table.insert(config_layers.moments, moments_dummy)
            table.insert(config_layers.gradientClips, gradientClip_dummy)
            table.insert(config_layers.weightClips, weightClip_other)
            table.insert(config_layers.weightRegConsts, weightRegConsts_dummy)
            table.insert(config_layers.initialRange, initialRange_dummy)
        end
    end

    for i=1, #gparams_current do
        grad_last[i] = gparams_current[i]:clone()
        grad_last[i]:fill(0)
    end
    return config_layers, grad_last
end

---------------------------------------
---- data IO relevant functions--------
---------------------------------------

function existfile(filename)
    local f=io.open(filename,"r")
    if f~=nil then io.close(f) return true else return false end
end

function load_filelist(fname)
    local data = file.read(fname)
    data = stringx.replace(data,'\n',' ')
    data = stringx.split(data)
    local imglist_ind = {}
    for i=1, #data do
        imglist_ind[i] = stringx.split(data[i],'.')[1]
    end
    return imglist_ind
end

function build_vocab(data, thresh, IDX_singleline, IDX_includeEnd)
    if IDX_singleline == 1 then
        data = stringx.split(data,'\n')
    else
        data = stringx.replace(data,'\n', ' ')
        data = stringx.split(data)
    end
    local countWord = {}
    for i=1, #data do
        if countWord[data[i]] == nil then
            countWord[data[i]] = 1
        else
            countWord[data[i]] = countWord[data[i]] + 1
        end
    end
    local vocab_map_ = {}
    local ivocab_map_ = {}
    local vocab_idx = 0
    if IDX_includeEnd==1 then
        vocab_idx = 1
        vocab_map_['NA'] = 1
        ivocab_map_[1] = 'NA'
    end

    for i=1, #data do
        if vocab_map_[data[i]]==nil then
            if countWord[data[i]]>=thresh then
                vocab_idx = vocab_idx+1
                vocab_map_[data[i]] = vocab_idx
                ivocab_map_[vocab_idx] = data[i]
                --print(vocab_idx..'-'.. data[i] ..'--'.. countWord[data[i]])
            else
                vocab_map_[data[i]] = vocab_map_['NA']
            end
        end 
    end
    vocab_map_['END'] = -1
    return vocab_map_, ivocab_map_, vocab_idx
end

function load_visualqadataset(opt, dataType, manager_vocab)
    -- Change it to your path. 
    -- local path_imglist = 'datasets/coco_dataset/allimage2014'
    -- All COCO images.
   
    -- VQA question/answer txt files.
    -- Download data_vqa_feat.zip and data_vqa_txt.zip and decompress into this folder
    local path_dataset = opt.datapath
    --local vgg_image_path = '/home/nfitz/data/VQA/images/'
    local vgg_image_path = '/home/nfitz/hdd2/data/VQA/images/'
    local vgg_feat_path = '/home/nfitz/data/VQA/images/'
    
    local prefix = 'coco_' .. dataType 
    local filename_question = paths.concat(path_dataset, prefix .. '_' .. opt.inputrep .. '.txt')
    local filename_answer = paths.concat(path_dataset, prefix .. '_answer.txt')
    local filename_imglist = paths.concat(path_dataset, prefix .. '_imglist.txt')
    local filename_allanswer = paths.concat(path_dataset, prefix .. '_allanswer.txt')
    local filename_choice = paths.concat(path_dataset, prefix .. '_choice.txt')
    local filename_choice_tag = paths.concat(path_dataset, prefix .. '_choice_tag.txt')
    local filename_choice_map = paths.concat(path_dataset, prefix .. '_choice_map.txt')
    local filename_question_type = paths.concat(path_dataset, prefix .. '_question_type.txt')
    local filename_answer_type = paths.concat(path_dataset, prefix .. '_answer_type.txt')
    local filename_questionID = paths.concat(path_dataset, prefix .. '_questionID.txt')

    if existfile(filename_allanswer) then
        data_allanswer = file.read(filename_allanswer)
        data_allanswer = stringx.split(data_allanswer,'\n')
    end
    if existfile(filename_choice) then
        data_choice = file.read(filename_choice)
        data_choice = stringx.split(data_choice, '\n')
    end
    if existfile(filename_choice_tag) then
        data_choice_tag = file.read(filename_choice_tag)
        data_choice_tag = stringx.split(data_choice_tag, '\n')
    end
    data_choice_map = {}
    if existfile(filename_choice_map) then
        data_choice_map_lines = file.read(filename_choice_map)
        data_choice_map_lines = stringx.split(data_choice_map_lines, '\n')

        for i, line in pairs(data_choice_map_lines) do
            if line ~= '' then
                local split = stringx.split(line, "\t")
                data_choice_map[split[1]] = split[2] 
            end
        end
    end
    if existfile(filename_question_type) then
        data_question_type = file.read(filename_question_type)
        data_question_type = stringx.split(data_question_type,'\n')
    end
    if existfile(filename_answer_type) then
        data_answer_type = file.read(filename_answer_type)
        data_answer_type = stringx.split(data_answer_type, '\n')
    end
    if  existfile(filename_questionID) then
        data_questionID = file.read(filename_questionID)
        data_questionID = stringx.split(data_questionID,'\n')
    end

    local data_answer
    local data_answer_split
    if existfile(filename_answer) then
        print("Load answer file = " .. filename_answer)
        data_answer = file.read(filename_answer)
        data_answer_split = stringx.split(data_answer,'\n')
    end

    print("Load question file = " .. filename_question)
    local data_question = file.read(filename_question)
    local data_question_split = stringx.split(data_question,'\n')
    local manager_vocab_ = {}

    if manager_vocab == nil then
        local vocab_map_answer, ivocab_map_answer, nvocab_answer = build_vocab(data_answer, opt.thresh_answerword, 1, 0)
        local vocab_map_question, ivocab_map_question, nvocab_question = build_vocab(data_question,opt.thresh_questionword, 0, 1)
        print(' no.vocab_question=' .. nvocab_question.. ', no.vocab_answer=' .. nvocab_answer)
        manager_vocab_ = {vocab_map_answer=vocab_map_answer, ivocab_map_answer=ivocab_map_answer, vocab_map_question=vocab_map_question, ivocab_map_question=ivocab_map_question, nvocab_answer=nvocab_answer, nvocab_question=nvocab_question}
    else
        manager_vocab_ = manager_vocab
    end

    local imglist = load_filelist(filename_imglist)
    local nSample = #imglist
    -- We can choose to run the first few answers.
    if nSample > #data_question_split then
        nSample = #data_question_split
    end

    -- Answers.
    local x_answer = torch.zeros(nSample):fill(-1)
    if opt.multipleanswer == 1 then
        x_answer = torch.zeros(nSample, 10)
    end
    local x_answer_num = torch.zeros(nSample)

    -- Convert words in answers and questions to indices into the dictionary.
    local x_question = torch.zeros(nSample, opt.seq_length)
    local x_seq_mask = torch.zeros(nSample, opt.seq_length)
    local x_seq_length = torch.zeros(nSample)
    local numNoAnswer = 0
    for i = 1, nSample do
        local words = stringx.split(data_question_split[i])
        x_seq_length[i] = math.min(#words, opt.seq_length)
        -- Answers
        --if existfile(filename_answer) then
        --    local answer = data_answer_split[i]
        --    if manager_vocab_.vocab_map_answer[answer] == nil then
        --        numNoAnswer = numNoAnswer + 1
        --        x_answer[i] = -1
        --    else
        --        x_answer[i] = manager_vocab_.vocab_map_answer[answer]
        --    end
        --end
        if existfile(filename_allanswer) then
            local answerline = data_allanswer[i]
            counts = {}
            for _, a in pairs(stringx.split(answerline, ',')) do
                a_id = manager_vocab_.vocab_map_answer[a]
                if a_id ~= nil then
                    if counts[a_id] == nil then
                        counts[a_id] = 0
                    end
                    counts[a_id] = counts[a_id] + 1
                end
            end

            local answer = -1
            local max_count = 0
            for a, count in pairs(counts) do
                if count > max_count then
                    answer = a
                    max_count = count
                end
            end
            x_answer[i] = answer
            if answer == -1 then
                numNoAnswer = numNoAnswer+1
            end
        elseif existfile(filename_answer) then
            local answer = data_answer_split[i]
            if manager_vocab_.vocab_map_answer[answer] == nil then
                numNoAnswer = numNoAnswer + 1
                x_answer[i] = -1
            else
                x_answer[i] = manager_vocab_.vocab_map_answer[answer]
            end
        end
 
        -- Questions
        for j = 1, opt.seq_length do
            if j <= #words then
                if manager_vocab_.vocab_map_question[words[j]] == nil then
                    x_question[{i, j}] = 1 
                else
                    x_question[{i, j}] = manager_vocab_.vocab_map_question[words[j]]
                end
                x_seq_mask[{i, j}] = 1
            else
                --x_question[{i, j}] = manager_vocab_.vocab_map_question['END']
                x_question[{i,j}] = torch.random(1, manager_vocab_.nvocab_question)
            end
        end
    end
    
    ---------------------------
    -- start loading features -
    ---------------------------
    local featureMap = {}
    local loading_spec = {
        trainval2014 = { train = true, val = true, test = false },
        trainval2014_train = { train = true, val = true, test = false },
        trainval2014_val = { train = false, val = true, test = false },
        train2014 = { train = true, val = false, test = false },
        val2014 = { train = false, val = true, test = false },
        test2015 = { train = false, val = false, test = true }
    }
    loading_spec['test-dev2015'] = { train = false, val = false, test = true }
 
    imagefileMap = {}
    if opt.trainvgg == 1 then
        local imagedir_prefixSet = {
            train = paths.concat(vgg_image_path, 'train2014', 'train2014_vgg_preproc'),
            val = paths.concat(vgg_image_path, 'val2014', 'val2014_vgg_preproc'),
            test = paths.concat(vgg_image_path, 'test2015', 'test2015_vgg_preproc')
        }

        for k, image_prefix in pairs(imagedir_prefixSet) do
            if loading_spec[dataType][k] then
                for file in lfs.dir(image_prefix) do
                    if stringx.endswith(file, ".t7") then
                        name = stringx.split(file, '.')[1]
                        --name = stringx.replace(name, '_cropped', '')
                        imagefileMap[name] = paths.concat(image_prefix, file)
                    end
                end
            end
        end
    elseif opt.vfeat == 'googlenetFC' then
        local featName = 'googlenetFCdense'
        print(featName)
        -- Possible combinations of data loading
        local feature_prefixSet = {
            train = paths.concat(path_dataset, 'coco_train2014_' .. featName), 
            val = paths.concat(path_dataset, 'coco_val2014_' .. featName),
            test = paths.concat(path_dataset,'coco_test2015_' .. featName)
        }

        for k, feature_prefix in pairs(feature_prefixSet) do
            -- Check if we need to load this dataset.
            if loading_spec[dataType][k] then
                local feature_imglist = torch.load(feature_prefix ..'_imglist.dat')
                local featureSet = torch.load(feature_prefix ..'_feat.dat')
                for i = 1, #feature_imglist do
                    local feat_in = torch.squeeze(featureSet[i])
                    featureMap[feature_imglist[i]] = feat_in
                end
            end
        end
    elseif opt.vfeat == 'vgg16' then
        local feature_prefixSet = {
            train = paths.concat(vgg_feat_path, 'train2014'),
            val = paths.concat(vgg_feat_path, 'val2014'),
            test = paths.concat(vgg_feat_path, 'test2015')
        }
        for k, feature_prefix in pairs(feature_prefixSet) do
            if loading_spec[dataType][k] then
                for file in lfs.dir(feature_prefix) do
                    if stringx.endswith(file, "_vggTop.t7") then
                        local name = stringx.replace(file, "_vggTop.t7", "")
                        local featureSet = torch.squeeze(torch.load(paths.concat(feature_prefix, file)).data)
                        featureMap[name] = featureSet
                    end
                end
            end
        end
    else
        error('Unrecognized feature spec: ' .. opt.vfeat)
    end

    collectgarbage()
    -- Return the state.
    local _state = {
        x_question = x_question, 
        x_seq_mask = x_seq_mask,
        x_seq_length = x_seq_length,
        x_answer = x_answer, 
        x_answer_num = x_answer_num, 
        featureMap = featureMap, 
        imagefileMap = imagefileMap,
        data_question = data_question_split,
        data_answer = data_answer_split, 
        imglist = imglist, 
        path_imglist = path_imglist, 
        data_allanswer = data_allanswer, 
        data_choice = data_choice,
        data_choice_tag = data_choice_tag, 
        data_choice_map = data_choice_map,
        data_question_type = data_question_type, 
        data_answer_type = data_answer_type, 
        data_questionID = data_questionID
    }
    
    return _state, manager_vocab_
end

--------------------------------------------
-- training relevant code
--------------------------------------------
function save_model(opt, manager_vocab, context, path)
    print('saving model ' .. path)
    local d = {}
    d.paramx = context.paramx:float()
    d.manager_vocab = manager_vocab
    d.stat = stat
    d.config_layers = config_layers
    d.opt = opt

    torch.save(path, d)
end

function bagofword(manager_vocab, x_seq)
-- turn the list of word index into bag of word vector
    local outputVector = torch.zeros(manager_vocab.nvocab_question)
    for i= 1, x_seq:size(1) do
        if x_seq[i] ~= manager_vocab.vocab_map_question['END'] then    
            outputVector[x_seq[i]] = 1
        else
            break
        end
    end
    return outputVector
end

function add_count(t, ...) 
    -- Input: table for counting, k1, v1, k2, v2, k3, v3
    -- Output: t[k1] += (v1, 1), t[k2] += (v2, 1), etc.
    local args = { ... }
    local i = 1
    while i < #args do
        local k = args[i]
        local v = args[i + 1]
        if t[k] == nil then 
            t[k] = { v, 1 } 
        else
            t[k][1] = t[k][1] + v
            t[k][2] = t[k][2] + 1
        end
        i = i + 2
    end
end

function compute_accuracy(t)
    local res = { }
    for k, v in pairs(t) do
        res[k] = v[1] / v[2]
    end
    return res
end

function evaluate_answer(state, manager_vocab, pred_answer, pred_answer_multi, selectIDX)
-- testing case for the VQA dataset
    selectIDX = selectIDX or torch.range(1, state.x_answer:size(1))
    local pred_answer_word = {}
    local gt_answer_word = state.data_answer
    local gt_allanswer = state.data_allanswer

    local perfs = { } 
    local count_question_type = {}
    local count_answer_type = {}

    for sampleID = 1, selectIDX:size(1) do
        local i = selectIDX[sampleID]

        -- Prediction correct. 
        if manager_vocab.ivocab_map_answer[pred_answer[i]]== gt_answer_word[i] then
            add_count(perfs, "most_freq", 1)
        else
            add_count(perfs, "most_freq",0)
        end

        -- Estimate using the standard criteria (min(#correct match/3, 1))
        -- Also estimate the mutiple choice case.
        local question_type = state.data_question_type[i]
        local answer_type = state.data_answer_type[i]

        local word_pred_answer_multiple = manager_vocab.ivocab_map_answer[pred_answer_multi[i]]
        local word_pred_answer_openend = manager_vocab.ivocab_map_answer[pred_answer[i]]

        -- Compare the predicted answer with all gt answers from humans.
        if gt_allanswer then
            local answers = stringx.split(gt_allanswer[i], ',')
            -- The number of answers matched with human answers.
            local count_curr_openend = 0
            local count_curr_multiple = 0
            for j = 1, #answers do
                count_curr_openend = count_curr_openend + (word_pred_answer_openend == answers[j] and 1 or 0)
                count_curr_multiple = count_curr_multiple + (word_pred_answer_multiple == answers[j] and 1 or 0)
            end

            local increment = math.min(count_curr_openend * 1.0/3, 1.0)
            add_count(perfs, "openend_overall", increment, 
                             "openend_q_" .. question_type, increment, 
                             "openend_a_" .. answer_type, increment)

            increment = math.min(count_curr_multiple * 1.0/3, 1.0)
            add_count(perfs, "multiple_overall", increment, 
                             "multiple_q_" .. question_type, increment, 
                             "multiple_a_" .. answer_type, increment)
        end
    end

    -- Compute accuracy
    return compute_accuracy(perfs)
end

function outputJSONanswer(state, manager_vocab, pred, pred_multi, file_json, choice)
    -- Dump the prediction result to csv file
    local f_json = io.open(file_json,'w')
    f_json:write('[')
    
    for i = 1, pred:size(1) do
        local word_pred_answer_multiple = manager_vocab.ivocab_map_answer[pred_multi[i]]
        local word_pred_answer_openend = manager_vocab.ivocab_map_answer[pred[i]]
        local answer_pred = word_pred_answer_openend
        if choice == 1 then
            answer_pred = word_pred_answer_multiple
        end
    
        local questionID = state.data_questionID[i]
        f_json:write('{"answer": "' .. answer_pred .. '","question_id": ' .. questionID .. '}')
        if i< pred:size(1) then
            f_json:write(',')
        end
    end
    f_json:write(']')
    f_json:close()

end

function new_batch(opt, manager_vocab) 
    local batch = {}
    batch = {}
    batch.IDXset_batch = torch.zeros(opt.batchsize)
    batch.target = torch.zeros(opt.batchsize)
    batch.word_idx = torch.zeros(opt.batchsize, opt.seq_length)
    batch.seq_mask = torch.zeros(opt.batchsize, opt.seq_length)
    return batch
end

function fill_batch(idx, currBatch, randIDX, state, opt, updateIDX)
    local n = state.x_question:size(1)
    local randIDX = torch.randperm(n)
    if updateIDX == 'test' or updateIDX == 'val' then
        randIDX = torch.range(1, n)
    end

    local nSample_batch = 0
    local dataset = nil
    if state.dataset ~= nil then
        dataset = state.dataset
    else
        dataset = {}
        dataset.size = n
        dataset.batches = {}
        dataset.batchSize = opt.batchsize
    end

    local nBatch = 1
    for iii = 1, n do
        if currBatch == nil then
            if dataset.batches[nBatch] == nil then
                currBatch = new_batch(opt, manager_vocab)
            else
                currBatch = dataset.batches[nBatch]
            end
        end

        local i = randIDX[iii]

        local first_answer = -1
        if updateIDX~='test' then
            first_answer = state.x_answer[i]
        end
        if first_answer == -1 and updateIDX == 'train' then
            --skip the sample with NA answer
        else
            nSample_batch = nSample_batch + 1
            currBatch.IDXset_batch[nSample_batch] = i
            if updateIDX ~= 'test' then
                currBatch.target[nSample_batch] = state.x_answer[i]
            end

            local filename = state.imglist[i]--'COCO_train2014_000000000092'

            currBatch.word_idx[nSample_batch] = state.x_question[i]
            currBatch.seq_mask[nSample_batch] = state.x_seq_mask[i]

            local feat_visual = nil
            if opt.trainvgg ~= 1 then
                feat_visual = state.featureMap[filename]:clone()
                currBatch.featBatch_visual[nSample_batch] = feat_visual:clone()
            else
                local file = state.imagefileMap[filename]
                local im = torch.load(file)
                currBatch.featBatch_visual[nSample_batch] = im.image:clone()
            end
                
            while iii == state.x_question:size(1) and nSample_batch< opt.batchsize do
                -- padding the extra sample to complete a batch for training
                nSample_batch = nSample_batch+1
                currBatch.IDXset_batch[nSample_batch] = i
                currBatch.target[nSample_batch] = first_answer
                currBatch.word_idx[nSample_batch] = state.x_question[i]
                currBatch.seq_mask[nSample_batch] = state.x_seq_mask[i]
                if opt.trainvgg ~= 1 then
                    currBatch.featBatch_visual[nSample_batch] = feat_visual:clone()
                else
                    local file = state.imagefileMap[filename]
                    local im = torch.load(file)
                    currBatch.featBatch_visual[nSample_batch] = im.image:clone()
                end
            end 
            if nSample_batch == opt.batchsize then                
                if dataset.batches[nBatch] == nil then
                    table.insert(dataset.batches, currBatch)
                end
                nBatch = nBatch+1
                nSample_batch = 0
                currBatch = nil
            end
        end
    end

    -- put all on GPU
    for _, batch in pairs(dataset.batches) do
        batch.IDXset_batch = batch.IDXset_batch:cuda()
        batch.target = batch.target:cuda()
        batch.word_idx = batch.word_idx:cuda()
        batch.featBatch_visual = batch.featBatch_visual:cuda()
        batch.seq_mask = batch.seq_mask:cuda()
    end

    print(string.format('make_batches took: %.2f (%d batches)', os.clock() - start_time, #dataset.batches))

    return dataset
end -- make_batches

function train_epoch(opt, state, manager_vocab, context, updateIDX)
    -- Dump context to the local namespace.
    local model = context.model
    local criterion = context.criterion
    local paramx = context.paramx
    local paramdx = context.paramdx
    local params_current = context.params_current
    local gparams_current = context.gparams_current
    local config_layers = context.config_layers
    local grad_last = context.grad_last
 
    local loss = 0.0
    local batch_loss = 0.0
    local pred_answer = torch.zeros(state.x_question:size(1)):cuda()
    local pred_answer_multi = torch.zeros(state.x_question:size(1)):cuda()

    local count_batch = 0
    local nBatch = 0

    if updateIDX == 'train' then
        model:training()
    else
        model:evaluate()
    end

    local batch = {}
    batch = {}
    batch.IDXset_batch = torch.zeros(opt.batchsize):cuda()
    batch.target = torch.zeros(opt.batchsize):cuda()
    batch.word_idx = torch.zeros(opt.batchsize, opt.seq_length):cuda()
    batch.seq_mask = torch.zeros(opt.batchsize, opt.seq_length):cuda()
    if opt.trainvgg == 1 then
        batch.featBatch_visual = torch.zeros(opt.batchsize, 3, 224, 224):cuda()
    else
        batch.featBatch_visual = torch.zeros(opt.batchsize, opt.vdim):cuda()
    end
 
    local start_epoch = os.clock()
    local last_tick = start_epoch
    local n = state.x_question:size(1)

    local randIDX = torch.randperm(n)
    if updateIDX == 'test' then
        randIDX = torch.range(1, n)
    end

    local idx = 1

    while idx < n do
        idx = fill_batch(idx, batch, randIDX, state, opt, updateIDX)

        if opt.method == 'BOW' then
            input = batch.featBatch_word
        elseif opt.method == 'BOWIMG' then
            input = {batch.featBatch_visual, batch.seq_mask, batch.word_idx}
        elseif opt.method == 'IMG' then
            input = batch.featBatch_visual
        else 
            print('error baseline method \n')
        end

        local output = model:forward(input)
        local err = criterion:forward(output, batch.target)
        local prob_batch = output:float()

        loss = loss + err

        -- Compute accuracy for multiple choices.
        if updateIDX ~= 'train' or opt.test_during_train==1 then
            local y_max, i_max = torch.max(prob_batch,2)
            i_max = torch.squeeze(i_max)
            for j = 1, opt.batchsize do
                local idx = batch.IDXset_batch[j]
                if idx <= state.x_question:size(1) then
                    local choices = stringx.split(state.data_choice[idx], ',')
                    local score_choices = torch.zeros(#choices):fill(-1000000)
                    for n = 1, #choices do
                        local IDX_pred = manager_vocab.vocab_map_answer[choices[n]]
                        if IDX_pred ~= nil then
                            local score = prob_batch[{j, IDX_pred}]
                            if score ~= nil then
                                score_choices[n] = score
                            end
                        end
                    end
                    local val_max, IDX_max = torch.max(score_choices, 1)
                    pred_answer_multi[idx] = manager_vocab.vocab_map_answer[choices[IDX_max[1]]]
                    if opt.constrainpred == 1 then
                        local choice_tag = state.data_choice_tag[idx]
                        local choice_str = state.data_choice_map[choice_tag]
                        if choice_str ~= nil then
                            local choices = stringx.split(state.data_choice_map[choice_tag], ',')
                            score_choices = torch.zeros(#choices):fill(-10000)
                            for n = 1, #choices do
                                IDX_pred = manager_vocab.vocab_map_answer[choices[n]]
                                if IDX_pred ~= nil then
                                    score = prob_batch[{j, IDX_pred}]
                                    if score ~= nil then
                                        score_choices[n] = score
                                    end
                                end
                            end
                            val_max, IDX_max = torch.max(score_choices, 1)
                            pred_answer[idx] = manager_vocab.vocab_map_answer[choices[IDX_max[1]]]
                        else
                            pred_answer[idx] = i_max[j]
                        end
                    else
                        pred_answer[idx] = i_max[j]
                    end
                end
            end
        end

        --------------------backforward pass
        if updateIDX == 'train' then
            model:zeroGradParameters()
            local df = criterion:backward(output, batch.target)
            local df_model = model:backward(input, df)

            -------------Update the params of baseline softmax---
            if opt.uniformLR ~= 1 then
                for i=1, #params_current do
                    local gnorm = gparams_current[i]:norm()
                    if config_layers.gradientClips[i]>0 and gnorm > config_layers.gradientClips[i] then
                        gparams_current[i]:mul(config_layers.gradientClips[i]/gnorm)
                    end

                    grad_last[i]:mul(config_layers.moments[i])
                    local tmp = torch.mul(gparams_current[i],-config_layers.lr_rates[i])
                    grad_last[i]:add(tmp)
                    params_current[i]:add(grad_last[i])
                    if config_layers.weightRegConsts[i]>0 then
                        local a = config_layers.lr_rates[i] * config_layers.weightRegConsts[i]
                        params_current[i]:mul(1-a)
                    end
                    local pnorm = params_current[i]:norm()
                    if config_layers.weightClips[i]>0 and pnorm > config_layers.weightClips[i] then
                        params_current[i]:mul(config_layers.weightClips[i]/pnorm)
                    end
                end
            else
                local norm_dw = paramdx:norm()
                if norm_dw > opt.max_gradientnorm then
                    local shrink_factor = opt.max_gradientnorm / norm_dw
                    paramdx:mul(shrink_factor)
                end
                paramx:add(g_paramdx:mul(-opt.lr))
            end
        end

        --batch finished

        nBatch = nBatch + 1
        if nBatch % 100 == 0 then
            local tick = os.clock()
            print(string.format("100 batches took: %.2f", tick - last_tick))
            last_tick = tick
        end
    
        count_batch = count_batch+1
        if count_batch == 120 then
            collectgarbage()
            count_batch = 0
        end

    end

    -- 1 epoch finished
    local perfs = nil
    if updateIDX ~= 'train' or opt.test_during_train==1 then
        if updateIDX~='test' then
            local gtAnswer = state.x_answer:clone():cuda()
            local correctNum = torch.sum(torch.eq(pred_answer, gtAnswer))
            acc = correctNum*1.0/pred_answer:size(1)
        else
            acc = -1
        end
        print(updateIDX ..': acc (mostFreq) =' .. acc)
        if updateIDX ~= 'test' and state.data_allanswer ~= nil then
            -- using the standard evalution criteria of QA virginiaTech
            perfs = evaluate_answer(state, manager_vocab, pred_answer, pred_answer_multi)
            print(updateIDX .. ': acc.match mostfreq = ' .. perfs.most_freq)
            print(updateIDX .. ': acc.dataset (OpenEnd) =' .. perfs.openend_overall)
            print(updateIDX .. ': acc.dataset (MultipleChoice) =' .. perfs.multiple_overall)
            -- If you want to see more statistics. do the following:
            -- print(perfs)
        end
        print(updateIDX .. ' loss=' .. loss/nBatch)
    end
    print(string.format('epoch took: %.2fs', os.clock()-start_epoch))
    return pred_answer, pred_answer_multi, perfs
end

