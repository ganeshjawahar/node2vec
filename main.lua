--[[

Node2Vec

]]--

require 'torch'
require 'io'
require 'nn'
require 'nnx'
require 'optim'
require 'os'
require 'xlua'
require 'lmdb'
Threads = require 'threads'
include('node2vec.lua')

cmd = torch.CmdLine()
-- data
cmd:option('-network_file', 'data/author_citation/author_citation_network', 'edge list for the network')
cmd:option('-labels_file', 'data/author_citation/author_labels', 'labels for each node in the network')
-- weight model
cmd:option('-use_weight', 1, 'should we capture the weight info?')
cmd:option('-m1_layers', '200,200', 'hidden layers specs. for the model')
cmd:option('-m1_batch_size', 10, 'number of sequences to train on in parallel')
-- link model
cmd:option('-use_link', 1, 'should we capture the link info?')
cmd:option('-m2_layers', '200,200', 'hidden layers specs. for the model')
cmd:option('-m2_batch_size', 10, 'number of sequences to train on in parallel')
cmd:option('-use_hsm', 1, 'use hierarchical softmax (yes = 1) or inefficient full softmax')
cmd:option('-hsm_bins', 100, 'hsm bin size')
-- label model
cmd:option('-use_label', 1, 'should we capture the label info?')
cmd:option('-m3_layers', '200,200', 'hidden layers specs. for the model')
cmd:option('-m3_batch_size', 10, 'number of sequences to train on in parallel')
-- commons
cmd:option('-embed_size', 128, 'size of the node embedding')
cmd:option('-learning_rate', 0.01, 'learning rate')
cmd:option('-max_epochs', 25, 'number of full passes through the training data')
cmd:option('-gpu', 1, '1=use gpu; 0=use cpu;')
cmd:option('-params_init', 0.08, 'initialize the parameters of the model from a uniform distribution')
cmd:option('-reg', 1e-4, 'regularization parameter l2-norm')
cmd:option('-use_reg', 0, 'use l2-regularization (1 = yes)')
cmd:option('-pre', 'node_', 'prefix for the output checkpoint')
cmd:option('-num_threads', 2, 'no. of asynchronous threads to use for asgd')
cmd:option('-async', 1, 'use asgd (1 = yes)')
cmd:option('-use_multi_label', 1, 'use multi-label classifier')

params = cmd:parse(arg)

if params.gpu == 1 then
	require 'cunn'
	require 'cutorch'
	require 'cunnx'
end

model = Node2Vec(params)
if params.async == 0 then
	model:train()
else
	model:train_async()
end