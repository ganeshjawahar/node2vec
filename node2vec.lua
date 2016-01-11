local Node2Vec = torch.class("Node2Vec")

function Node2Vec:__init(config)
	self.network_file = config.network_file
	self.labels_file = config.labels_file
	self.use_weight = config.use_weight
	self.m1_layers = config.m1_layers
	self.use_link = config.use_link
	self.m2_layers = config.m2_layers
	self.use_label = config.use_label
	self.m3_layers = config.m3_layers
	self.embed_size = config.embed_size
	self.learning_rate = config.learning_rate
	self.batch_size = config.batch_size
	self.max_epochs = config.max_epochs
	self.gpu = config.gpu
	self.params_init = config.params_init
	self.reg = config.reg
	self.pre = config.pre
	self.num_threads = config.num_threads
	self.db = lmdb.env{Path = './'..self.pre..'n2vDB', Name = self.pre..'n2vDB'}

	-- create the vocabulary
	self:create_vocab()

	-- create the model
	self:create_model()

	-- create the batches
	-- self:create_batches()
end

function Node2Vec:create_batches()
	print('creating batches...')
	local start = sys.clock()
	self.db:open()
	local writer = self.db:txn()
	if self.use_weight == 1 then
		local sample_idx = 0
		print('creating samples for m1...')
		xlua.progress(1, self.edge_count)
		c = 0
		for line in io.lines(self.network_file) do
			line = string.gsub(line, '\n', '')
			local content = string.split(line, '\t')
			local weight = nil
			if #content > 2 then
				weight = tonumber(content[3]) / self.total_weight
			else
				weight = 1 / self.total_weight
			end
			local tensor_a = torch.IntTensor{self.node_2_id[content[1]]}
			local tensor_b = torch.IntTensor{self.node_2_id[content[2]]}
			local label = torch.Tensor{weight}
			if self.gpu == 1 then
				tensor_a = tensor_a:cuda()
				tensor_b = tensor_b:cuda()
				label = label:cuda()
			end
			sample_idx = sample_idx + 1
			writer:put('m1_'..sample_idx, {{tensor_a, tensor_b}, label})
			if sample_idx % 300000 == 0 then
				writer:commit()
				writer = self.db:txn()
				collectgarbage()
			end
			if sample_idx % 200 == 0 then
				xlua.progress(sample_idx, self.edge_count)
			end			
			c = c + 1
			if c == 10000 then
				-- break
			end
		end
		xlua.progress(self.edge_count, self.edge_count)
		print(sample_idx..' samples created.')
		writer:put('m1_size', sample_idx)
	end
	if self.use_link == 1 then
		local sample_idx = 0
		print('creating samples for m2...')
		xlua.progress(1, self.edge_count)
		c = 0
		for line in io.lines(self.network_file) do
			line = string.gsub(line, '\n', '')
			local content = string.split(line, '\t')
			local tensor_a = torch.IntTensor{self.node_2_id[content[1]]}
			local tensor_b = torch.IntTensor{self.node_2_id[content[2]]}
			if self.gpu == 1 then
				tensor_a = tensor_a:cuda()
				tensor_b = tensor_b:cuda()
			end
			sample_idx = sample_idx + 1
			writer:put('m2_'..sample_idx, {tensor_a, tensor_b})
			if sample_idx % 300000 == 0 then
				writer:commit()
				writer = self.db:txn()
				collectgarbage()
			end
			if sample_idx % 200 == 0 then
				xlua.progress(sample_idx, self.edge_count)
			end						
			c = c + 1
			if c == 10000 then
				-- break
			end
		end
		xlua.progress(self.edge_count, self.edge_count)
		print(sample_idx..' samples created.')
		writer:put('m2_size', sample_idx)
	end
	if self.use_label == 1 then
		local sample_idx = 0
		print('creating samples for m3...')
		xlua.progress(1, self.labelled_nodes_count)
		c = 0
		for line in io.lines(self.labels_file) do
			line = string.gsub(line, '\n', '')
			local content = string.split(line, '\t')
			local labs = string.split(content[2], ',')
			for i = 1, #labs do
				local input = torch.IntTensor{self.node_2_id[content[1]]}
				local label = torch.IntTensor{self.label_2_id[labs[i]]}
				if self.gpu == 1 then
					input = input:cuda()
					label = label:cuda()
				end
				sample_idx = sample_idx + 1
				writer:put('m3_'..sample_idx, {input, label})
				if sample_idx % 300000 == 0 then
					writer:commit()
					writer = self.db:txn()
					collectgarbage()
				end
				if sample_idx % 200 == 0 then
					xlua.progress(sample_idx, self.labelled_nodes_count)
				end	
			end			
			c = c + 1
			if c == 10000 then
				-- break
			end
		end
		xlua.progress(self.labelled_nodes_count, self.labelled_nodes_count)
		print(sample_idx..' samples created.')
		writer:put('m3_size', sample_idx)
	end
	collectgarbage()
	writer:commit()
	self.db:close()
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

function Node2Vec:train_async()
	print('training in hogwild fashion...')
	local start = sys.clock()
	self.db:open()
	local reader = self.db:txn(true)
	for epoch = 1, 5 do
		local epoch_start = sys.clock()
		if self.use_weight == 1 then
			local function threadedTrain(model, criterion, num_threads, reader, is_gpu, batch_size)
				Threads.serialization('threads.sharedserialize')
				local threads = Threads(
					num_threads,
					function()
						require 'nn'
						require 'cunn'
						require 'cutorch'
						require 'lmdb'
						include('node2vec.lua')
					end,
					function()
						local model = model:clone('weight', 'bias')
						local criterion = criterion:clone()
						local weights, dweights = model:parameters()

						function gupdate(input, label)
							local pred = model:forward(input)
							local loss = criterion:forward(pred, label)
							model:zeroGradParameters()
							local grads = criterion:updateGradInput(pred, label)
							model:updateGradInput(input, grads)
							model:accGradParameters(input, grads)
							for wi = 1, #dweights do
								dweights[wi] = dweights[wi]:div(#input)
							end
							loss = loss / #input
							return loss, dweights
						end
					end
				)
				local weights = model:parameters()
				local m1_size = reader:get('m1_size')
				local indices = torch.randperm(m1_size)
				local epoch_loss, epoch_iteration = 0, 0
				local m1_start = sys.clock()
				local num_batches = torch.floor(m1_size / batch_size)
				xlua.progress(1, num_batches)
				for i = 1, m1_size, batch_size do
					input, label = {}, torch.Tensor(batch_size)
					if is_gpu == 1 then label = label:cuda() end
					for j = i, math.min(i + batch_size - 1, m1_size) do
						local data = reader:get('m1_'..indices[j])
						table.insert(input, data[1])
						label[#input] = data[2]
					end
					threads:addjob(
						function(input, label)
							return gupdate(input, label)
						end,
						function(err, dweights)
							epoch_loss = epoch_loss + err
							epoch_iteration = epoch_iteration + 1
							for i = 1, #weights do
								weights[i]:add(-0.01, dweights[i])
							end
							if epoch_iteration % 5 == 0 then
								collectgarbage()
								xlua.progress(epoch_iteration, num_batches)
							end
							epoch_iteration = epoch_iteration + 1
						end,
						input,
						label
					)
					epoch_iteration = epoch_iteration + 1					
					if epoch_iteration > num_batches then break end
				end
				threads:synchronize()
				threads:terminate()
				xlua.progress(m1_size, m1_size)
				print(string.format("Done in %.2f minutes. Weight-Info Loss = %f\n",((sys.clock() - m1_start) / 60), (epoch_loss / epoch_iteration)))
			end
			threadedTrain(self.weight_model, self.weight_criterion, self.num_threads, reader, self.gpu, self.batch_size)
		end
		self:save_model(self.pre..epoch..'.t7')
		if epoch ~= 1 then
			os.execute('rm '..self.pre..(epoch-1)..'.t7')
		end
		print(string.format("Epoch %d Done in %.2f minutes.", epoch, (sys.clock() - epoch_start)/60))
	end
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

function Node2Vec:train()
	print('training...')
	local start = sys.clock()
	self.db:open()
	local reader = self.db:txn(true)
	for epoch = 1, self.max_epochs do
		local epoch_start = sys.clock()
		if self.use_weight == 1 then
			local m1_size = reader:get('m1_size')
			local indices = torch.randperm(m1_size)
			local m1_start = sys.clock()
			local epoch_loss, epoch_iteration = 0, 0
			local num_batches = torch.floor(m1_size / self.batch_size)
			xlua.progress(1, num_batches)
			for i = 1, m1_size, self.batch_size do
				self.m1_input, self.m1_label = {}, torch.Tensor(self.batch_size)
				if self.gpu == 1 then self.m1_label = self.m1_label:cuda() end
				for j = i, math.min(i + self.batch_size - 1, m1_size) do
					local data = reader:get('m1_'..indices[j])
					table.insert(self.m1_input, data[1])
					self.m1_label[#self.m1_input] = data[2]
				end
				local _, loss = optim.sgd(self.m1_feval, self.m1_params, self.m1_optim_state)
				epoch_loss = epoch_loss + loss[1]
				epoch_iteration = epoch_iteration + 1
				if epoch_iteration % 5 == 0 then
					xlua.progress(epoch_iteration, num_batches)
				end
				if epoch_iteration % 100 == 0 then
					collectgarbage()
				end
				if epoch_iteration > num_batches then break end
			end
			xlua.progress(num_batches, num_batches)
			print(string.format("Done in %.2f minutes. Weight-Info Loss = %f\n",((sys.clock() - m1_start) / 60), (epoch_loss / epoch_iteration)))
		end
		if self.use_link == 1 then
			local m2_size = reader:get('m2_size')
			local indices = torch.randperm(m2_size)
			local m2_start = sys.clock()
			local epoch_loss, epoch_iteration = 0, 0
			xlua.progress(1, m2_size)
			for i = 1, m2_size, self.batch_size do
				self.m2_input, self.m2_label = {}, {}
				for j = i, math.min(i + self.batch_size - 1, m2_size) do
					local data = reader:get('m2_'..indices[j])
					table.insert(self.m2_input, data)
					table.insert(self.m2_label, data[2])
				end
				local _, loss = optim.sgd(self.m2_feval, self.m2_params, self.m2_optim_state)
				epoch_loss = epoch_loss + loss[1]
				epoch_iteration = epoch_iteration + 1
				xlua.progress(i, m2_size)
				if epoch_iteration % 100 == 0 then
					collectgarbage()
				end
			end
			xlua.progress(m2_size, m2_size)
			print(string.format("Done in %.2f minutes. Link-Info Loss = %f\n",((sys.clock() - m2_start) / 60), (epoch_loss / epoch_iteration)))
		end
		if self.use_label == 1 then
			local m3_size = reader:get('m3_size')
			local indices = torch.randperm(m3_size)
			local m3_start = sys.clock()
			local epoch_loss, epoch_iteration = 0, 0
			xlua.progress(1, m3_size)
			for i = 1, m3_size, self.batch_size do
				self.m3_input, self.m3_label = {}, {}
				for j = i, math.min(i + self.batch_size - 1, m3_size) do
					local data = reader:get('m3_'..indices[j])
					table.insert(self.m3_input, data[1])
					table.insert(self.m3_label, data[2])
				end
				local _, loss = optim.sgd(self.m3_feval, self.m3_params, self.m3_optim_state)
				epoch_loss = epoch_loss + loss[1]
				epoch_iteration = epoch_iteration + 1
				xlua.progress(i, m3_size)
				if epoch_iteration % 100 == 0 then
					collectgarbage()
				end
			end
			xlua.progress(m3_size, m3_size)
			print(string.format("Done in %.2f minutes. Label-Info Loss = %f\n",((sys.clock() - m3_start) / 60), (epoch_loss / epoch_iteration)))
		end	
		self:save_model(self.pre..epoch..'.t7')
		if epoch ~= 1 then
			os.execute('rm '..self.pre..(epoch-1)..'.t7')
		end
		print(string.format("Epoch %d Done in %.2f minutes.", epoch, (sys.clock() - epoch_start)/60))
	end
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

function Node2Vec:save_model(file)
	local checkpoint = {}
	checkpoint.node_2_id = self.node_2_id
	checkpoint.id_2_node = self.id_2_node
	checkpoint.node_weights = self.node_vecs.weight
	torch.save(file, checkpoint)
end

function Node2Vec:create_model()
	print('creating model...')
	local start = sys.clock()
	self.node_vecs = nn.LookupTable(#self.id_2_node, self.embed_size)

	if self.use_weight == 1 then
		self.node_model = nn.Sequential()
		self.node_model:add(self.node_vecs:clone('weight', 'bias', 'gradWeight', 'gradBias'))
		local layers = string.split(self.m1_layers, ',')
		local input_size = self.embed_size
		for i = 1, #layers do
			self.node_model:add(nn.Linear(input_size, tonumber(layers[i])))
			input_size = tonumber(layers[i])
		end
		self._weight_model = nn.Sequential()
		self._weight_model:add(nn.ParallelTable())
		self._weight_model.modules[1]:add(self.node_model)
		self._weight_model.modules[1]:add(self.node_model:clone('weight', 'bias', 'gradWeight', 'gradBias'))
		self._weight_model:add(nn.MM(false, true))
		self._weight_model:add(nn.Sigmoid())
		
		self.weight_model = nn.Sequential()
		self.weight_model:add(nn.ParallelTable())		
		for i = 1, self.batch_size do
			self.weight_model.modules[1]:add(self._weight_model:clone('weight', 'bias', 'gradWeight', 'gradBias'))
		end
		self.weight_model:add(nn.JoinTable(1))
		self.weight_criterion = nn.BCECriterion()
		if self.gpu == 1 then 
			self.weight_model = self.weight_model:cuda()
			self.weight_criterion = self.weight_criterion:cuda()
		end

		self.m1_optim_state = {learningRate = self.learning_rate}
		self.m1_params, self.m1_grad_params = self.weight_model:getParameters()
		self.m1_input, self.m1_label = nil, nil
		self.m1_params:uniform(-1 * self.params_init, self.params_init)
		self.m1_feval = function(z)
			if z ~= self.m1_params then
				self.m1_params:copy(z)
			end
			self.m1_grad_params:zero()
			local loss = 0
			local pred = self.weight_model:forward(self.m1_input)
			local _loss = self.weight_criterion:forward(pred, self.m1_label)
			loss = loss + _loss
			local grads = self.weight_criterion:backward(pred, self.m1_label)
			self.weight_model:backward(self.m1_input, grads)
			self.m1_grad_params:div(#self.m1_input)
			return loss / #self.m1_input, self.m1_grad_params
		end
	end

	if self.use_link == 1 then
		self._link_model = nn.Sequential()
		self._link_model:add(self.node_vecs:clone('weight', 'bias', 'gradWeight', 'gradBias'))
		local layers = string.split(self.m2_layers, ',')
		local input_size = self.embed_size
		for i = 1, #layers do
			self._link_model:add(nn.Linear(input_size, tonumber(layers[i])))
			input_size = tonumber(layers[i])
		end		
		self.tree, self.root = self:create_frequency_tree(#self.id_2_node)
		self.link_model = nn.Sequential()
		self.link_model:add(nn.ParallelTable())
		self.link_model.modules[1]:add(self._link_model)
		self.link_model.modules[1]:add(nn.Identity())
		self.link_model:add(nn.SoftMaxTree(input_size, self.tree, self.root))
		self.link_criterion = nn.TreeNLLCriterion()
		if self.gpu == 1 then 
			self.link_model = self.link_model:cuda()
			self.link_criterion = self.link_criterion:cuda()
		end

		self.m2_optim_state = {learningRate = self.learning_rate}
		self.m2_params, self.m2_grad_params = self.link_model:getParameters()
		self.m2_params:uniform(-1 * self.params_init, self.params_init)
		self.m2_input, self.m2_label = nil, nil
		self.m2_feval = function(z)
			if z ~= self.m2_params then
				self.m2_params:copy(z)
			end
			self.m2_grad_params:zero()
			local loss = 0
			for i = 1, #self.m2_input do
				local pred = self.link_model:forward(self.m2_input[i])
				local _loss = self.link_criterion:forward(pred, self.m2_label[i])
				loss = loss + _loss
				local grads = self.link_criterion:backward(pred, self.m2_label[i])
				self.link_model:backward(self.m2_input[i], grads)
			end
			self.m2_grad_params:div(#self.m2_input)
			return loss / #self.m2_input, self.m2_grad_params
		end
	end

	if self.use_label == 1 then
		self.label_model = nn.Sequential()
		self.label_model:add(self.node_vecs:clone('weight', 'bias', 'gradWeight', 'gradBias'))
		local layers = string.split(self.m3_layers, ',')
		local input_size = self.embed_size
		for i = 1, #layers do
			self.label_model:add(nn.Linear(input_size, tonumber(layers[i])))
			input_size = tonumber(layers[i])
		end
		self.label_model:add(nn.Linear(input_size, #self.id_2_label))
		self.label_criterion = nn.CrossEntropyCriterion()
		if self.gpu == 1 then 
			self.label_model = self.label_model:cuda()
			self.label_criterion = self.label_criterion:cuda()
		end

		self.m3_optim_state = {learningRate = self.learning_rate}
		self.m3_params, self.m3_grad_params = self.label_model:getParameters()
		self.m3_params:uniform(-1 * self.params_init, self.params_init)		
		self.m3_input, self.m3_label = nil, nil
		self.m3_feval = function(z)
			if z ~= self.m3_params then
				self.m3_params:copy(z)
			end
			self.m3_grad_params:zero()
			local loss = 0
			for i = 1, #self.m3_input do
				local pred = self.label_model:forward(self.m3_input[i])
				local _loss = self.label_criterion:forward(pred, self.m3_label[i])
				loss = loss + _loss
				local grads = self.label_criterion:backward(pred, self.m3_label[i])
				self.label_model:backward(self.m3_input[i], grads)
			end
			self.m3_grad_params:div(#self.m3_input)
			loss = loss / #self.m3_input
			-- loss = loss + 0.5 * self.reg * self.m3_params:norm()^2
			return loss, self.m3_grad_params
		end
	end
		
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

function Node2Vec:create_vocab()
	print('creating vocab...')
	local start = sys.clock()
	self.id_2_node, self.node_2_id = {}, {}
	self.total_weight = 0
	self.edge_count = 0
	c = 0
	for line in io.lines(self.network_file) do
		line = string.gsub(line, '\n', '')
		local content = string.split(line, '\t')
		if self.node_2_id[content[1]] == nil then
			self.id_2_node[#self.id_2_node + 1] = content[1]
			self.node_2_id[content[1]] = #self.id_2_node
		end
		if self.node_2_id[content[2]] == nil then
			self.id_2_node[#self.id_2_node + 1] = content[2]
			self.node_2_id[content[2]] = #self.id_2_node
		end
		if #content > 2 then
			local weight = tonumber(content[3])
			self.total_weight = self.total_weight + weight
		else
			self.total_weight = self.total_weight + 1
		end	
		self.edge_count = self.edge_count + 1
		c = c + 1
		if c == 10000 then
			-- break
		end
	end
	print('#nodes = '..#self.id_2_node)
	print('total weight = '..self.total_weight)
	self.label_2_id, self.id_2_label = {}, {}
	self.labelled_nodes_count = 0
	c = 0
	for line in io.lines(self.labels_file) do
		line = string.gsub(line, '\n', '')
		local content = string.split(line, '\t')
		local labs = string.split(content[2], ',')
		for i = 1, #labs do
			if self.label_2_id[labs[i]] == nil then
				self.id_2_label[#self.id_2_label + 1] = labs[i]
				self.label_2_id[labs[i]] = #self.id_2_label
			end
		end		
		if self.node_2_id[content[1]] == nil then
			self.id_2_node[#self.id_2_node + 1] = content[1]
			self.node_2_id[content[1]] = #self.id_2_node
		end
		self.labelled_nodes_count = self.labelled_nodes_count + 1
		c = c + 1
		if c == 10000 then
			-- break
		end
	end	
	print('#labels = '..#self.id_2_label)
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

function Node2Vec:create_frequency_tree(node_count)
	freq_map = self:create_node_map(node_count)
	binSize = 100
	local ft = torch.IntTensor(freq_map)
	local vals, indices = ft:sort()
	local tree = {}
	local id = indices:size(1)
	function recursiveTree(indices)
		if indices:size(1) < binSize then
			id = id + 1
			tree[id] = indices
			return
		end
		local parents = {}
		for start = 1, indices:size(1), binSize do
			local stop = math.min(indices:size(1), start + binSize - 1)
			local bin = indices:narrow(1, start, stop - start + 1)
			assert(bin:size(1) <= binSize)
			id = id + 1
			table.insert(parents,id)
			tree[id] = bin
		end
		recursiveTree(indices.new(parents))
	end
	recursiveTree(indices)
	return tree, id
end

function Node2Vec:create_node_map(node_count)
	node_map = {}
	for i = 1, node_count do
		node_map[i] = 1
	end
	return node_map
end