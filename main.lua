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
cmd:option('-network_file', 'data/author_citation_network', 'edge list for the network')
cmd:option('-labels_file', 'data/author_labels', 'labels for each node in the network')
-- weight model
cmd:option('-use_weight', 1, 'should we capture the weight info?')
cmd:option('-m1_layers', '200,200', 'hidden layers specs. for the model')
-- link model
cmd:option('-use_link', 0, 'should we capture the link info?')
cmd:option('-m2_layers', '200,200', 'hidden layers specs. for the model')
-- label model
cmd:option('-use_label', 0, 'should we capture the label info?')
cmd:option('-m3_layers', '200,200', 'hidden layers specs. for the model')
-- commons
cmd:option('-embed_size', 128, 'size of the node embedding')
cmd:option('-learning_rate', 0.1, 'learning rate')
cmd:option('-batch_size', 30, 'number of sequences to train on in parallel')
cmd:option('-max_epochs', 100, 'number of full passes through the training data')
cmd:option('-gpu', 1, '1=use gpu; 0=use cpu;')
cmd:option('-params_init', 0.08, 'initialize the parameters of the model from a uniform distribution')
cmd:option('-reg', 1e-4, 'regularization parameter l2-norm')
cmd:option('-pre', 'node_', 'prefix for the output checkpoint')
cmd:option('-num_threads', 2, 'no. of asynchronous threads to use for asgd')
cmd:option('-async', 1, 'use asgd (1 = yes)')

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