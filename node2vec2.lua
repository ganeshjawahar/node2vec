local Node2Vec = torch.class("Node2Vec")

function Node2Vec:__init(config)
	self.network_file = config.network_file
	self.labels_file = config.labels_file
	self.use_weight = config.use_weight
	self.m1_layers = config.m1_layers
	self.m1_batch_size = config.m1_batch_size
	self.use_link = config.use_link
	self.m2_layers = config.m2_layers
	self.m2_batch_size = config.m2_batch_size
	self.m2_neg_samples = config.m2_neg_samples
	self.use_hsm = config.use_hsm
	self.hsm_bins = config.hsm_bins
	self.use_label = config.use_label
	self.m3_layers = config.m3_layers
	self.m3_batch_size = config.m3_batch_size
	self.embed_size = config.embed_size
	self.learning_rate = config.learning_rate
	self.max_epochs = config.max_epochs
	self.gpu = config.gpu
	self.params_init = config.params_init
	self.reg = config.reg
	self.pre = config.pre
	self.num_threads = config.num_threads
	self.use_reg = config.use_reg	
	self.use_multi_label = config.use_multi_label
	self.use_decay = config.use_decay
	self.decay_rate = config.decay_rate
	self.db = lmdb.env{Path = './'..self.pre..'_n2vDB', Name = self.pre..'_n2vDB'}
	self.db:open(); self.db:close();

	-- create the vocabulary
	self:create_vocab()

	-- create the model
	self:create_model()

	-- create the batches
	self:create_batches()
end

function Node2Vec:train()
	print('training...')
	local start = sys.clock()
	self.db:open()
	local reader = self.db:txn(true)
	for epoch = 1, self.max_epochs do
		local epoch_start = sys.clock()
		if self.use_link == 1 then
			local m2_size = reader:get('m2_size')
			local indices = torch.randperm(m2_size)
			local m2_start = sys.clock()
			local epoch_loss, epoch_iteration = 0, 0
			xlua.progress(1, m2_size)
			self.m2_optim_state.learningRate = self.learning_rate
			for i = 1, m2_size do
				local data = reader:get('m2_'..indices[i])
				self.m2_input, self.m2_label = {data[1], data[2]}, data[3]
				local _, loss = optim.sgd(self.m2_feval, self.m2_params, self.m2_optim_state)
				epoch_loss = epoch_loss + loss[1]
				epoch_iteration = epoch_iteration + 1
				if epoch_iteration % 5 == 0 then
					xlua.progress(epoch_iteration, m2_size)
					collectgarbage()
				end
				if self.use_decay == 1 then
					self.m2_optim_state.learningRate = self.m2_optim_state.learningRate * (1 - (i / m2_size))
				end
			end
			xlua.progress(m2_size, m2_size)
			print(string.format("Done in %.2f minutes. Link-Info Loss = %f\n",((sys.clock() - m2_start) / 60), (epoch_loss / epoch_iteration)))
		end
		if epoch % 5 == 0 then
			self:save_model(self.pre..epoch..'.t7')
		end
		--[[
		if epoch ~= 1 then
			os.execute('rm '..self.pre..(epoch-1)..'.t7')
		end
		]]--
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

function Node2Vec:create_batches()
	print('creating batches...')
	local start = sys.clock()
	self.db:open()
	local writer = self.db:txn()
	if self.use_link == 1 then
		local adj_list = {}
		for line in io.lines(self.network_file) do
			line = string.gsub(line, '\n', '')
			local content = string.split(line, '\t')
			if adj_list[self.node_2_id[content[1]]] == nil then adj_list[self.node_2_id[content[1]]] = {} end
			if adj_list[self.node_2_id[content[2]]] == nil then adj_list[self.node_2_id[content[2]]] = {} end
			table.insert(adj_list[self.node_2_id[content[1]]], self.node_2_id[content[2]])
			table.insert(adj_list[self.node_2_id[content[2]]], self.node_2_id[content[1]])
		end
		local sample_idx = 0
		print('creating batches for m2...')
		local num_batches = torch.floor((self.edge_count + (self.m2_neg_samples * #self.id_2_node)) / self.m2_batch_size)
		local input, input_context, label = {}, {}, torch.Tensor(self.m2_batch_size)
		xlua.progress(1, num_batches)
		for key, value in pairs(adj_list) do
			samples = self:get_samples(key, adj_list[key])
			for i, val in ipairs(samples) do
				table.insert(input, val[1])
				table.insert(input_context, val[2])
				label[#input] = val[3]
				if #input == self.m2_batch_size then
					sample_idx = sample_idx + 1
					writer:put('m2_'..sample_idx, {input, input_context, label:cuda()})
					if sample_idx % 50000 == 0 then
						writer:commit()
						writer = self.db:txn()
						collectgarbage()
					end
					input, input_context = nil, nil
					input, input_context = {}, {}
					if sample_idx % 5 == 0 then
						xlua.progress(sample_idx, num_batches)
					end
				end
			end
		end
		if #input ~= 0 then
			sample_idx = sample_idx + 1
			local cur_size = #input
			for j = cur_size + 1, self.m1_batch_size do
				table.insert(input, input[1])
				table.insert(input_context, input_context[1])
				label[j] = label[1]
			end
			writer:put('m2_'..sample_idx, {input, input_context, label:cuda()})
			input, input_context, label = nil, nil, nil
		end
		xlua.progress(num_batches, num_batches)
		writer:put('m2_size', sample_idx)
		adj_list = nil
		collectgarbage()
	end
	self.table = nil
	collectgarbage()
	writer:commit()
	self.db:close()
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

function Node2Vec:get_samples(node, neighbors)
	local samples = {}
	local input = torch.IntTensor{node}
	if self.gpu == 1 then input = input:cuda() end
	for i, c in ipairs(neighbors) do
		local context = torch.IntTensor{c}
		if self.gpu == 1 then context = context:cuda(); end
		table.insert(samples, {input, context, 1})
	end
	-- generate negative samples
	local i = 0
	while i < self.m2_neg_samples do
		neg_context = self.table[torch.random(self.table_size)]
		if node ~= neg_context and neighbors[neg_context] == nil then
			local context = torch.IntTensor{neg_context}
			if self.gpu == 1 then context = context:cuda(); end
			table.insert(samples, {input, context, -1})	
			i = i + 1
		end
	end
	return samples
end

function Node2Vec:create_model()
	print('creating model...')
	local start = sys.clock()
	self.node_vecs = nn.LookupTable(#self.id_2_node, self.embed_size)
	self.context_vecs = nn.LookupTable(#self.id_2_node, self.embed_size)

	if self.use_link == 1 then
		print('creating link model...')
		self._link_model = nn.Sequential()
		self._link_model:add(self.node_vecs:clone('weight', 'bias', 'gradWeight', 'gradBias'))
		local layers = string.split(self.m2_layers, ',')
		local input_size = self.embed_size
		for i = 1, #layers do
			self._link_model:add(nn.Linear(input_size, tonumber(layers[i])))
			input_size = tonumber(layers[i])
		end
		self.__link_model = nn.Sequential()
		self.__link_model:add(nn.ParallelTable())
		xlua.progress(1, self.m2_batch_size)
		for i = 1, self.m2_batch_size do
			self.__link_model.modules[1]:add(self._link_model:clone('weight', 'bias', 'gradWeight', 'gradBias'))
			if i % 10 == 0 then
				xlua.progress(i, self.m2_batch_size)
			end
		end
		xlua.progress(self.m2_batch_size, self.m2_batch_size)
		self.__link_model:add(nn.JoinTable(1))

		self._link_context_model = nn.Sequential()
		self._link_context_model:add(self.context_vecs:clone('weight', 'bias', 'gradWeight', 'gradBias'))
		input_size = self.embed_size
		for i = 1, #layers do
			self._link_context_model:add(nn.Linear(input_size, tonumber(layers[i])))
			input_size = tonumber(layers[i])
		end
		self.__link_context_model = nn.Sequential()
		self.__link_context_model:add(nn.ParallelTable())
		xlua.progress(1, self.m2_batch_size)
		for i = 1, self.m2_batch_size do
			self.__link_context_model.modules[1]:add(self._link_context_model:clone('weight', 'bias', 'gradWeight', 'gradBias'))
			if i % 10 == 0 then
				xlua.progress(i, self.m2_batch_size)
			end
		end
		xlua.progress(self.m2_batch_size, self.m2_batch_size)
		self.__link_context_model:add(nn.JoinTable(1))

		self.link_model = nn.Sequential()
		self.link_model:add(nn.ParallelTable())
		self.link_model.modules[1]:add(self.__link_model)
		self.link_model.modules[1]:add(self.__link_context_model)

		self.link_criterion = nn.CosineEmbeddingCriterion(0.5)
		self.link_criterion.sizeAverage = false
		if self.gpu == 1 then 
			self.link_model = self.link_model:cuda()
			self.link_criterion = self.link_criterion:cuda()
		end

		self.m2_optim_state = nil 
		if self.use_decay == 0 then
			self.m2_optim_state = {learningRate = self.learning_rate, alpha = self.decay_rate}
		else
			self.m2_optim_state = {learningRate = self.learning_rate}
		end
		self.m2_params, self.m2_grad_params = self.link_model:getParameters()
		self.m2_params:uniform(-1 * self.params_init, self.params_init)
		self.m2_input, self.m2_label = nil, nil
		self.m2_feval = function(z)
			if z ~= self.m2_params then
				self.m2_params:copy(z)
			end
			self.m2_grad_params:zero()
			local pred = self.link_model:forward(self.m2_input)
			local loss = self.link_criterion:forward(pred, self.m2_label)
			local grads = self.link_criterion:backward(pred, self.m2_label)
			self.link_model:backward(self.m2_input, grads)
			self.m2_grad_params:div(self.m2_batch_size)
			loss = loss / self.m2_batch_size
			if self.use_reg == 1 then
				loss = loss + 0.5 * self.reg * self.m2_params:norm()^2
			end
			return loss, self.m2_grad_params
		end
	end
end

function Node2Vec:create_vocab()
	print('creating vocab...')
	local start = sys.clock()
	self.id_2_node, self.node_2_id = {}, {}
	self.total_weight = 0
	self.edge_count = 0
	self.vocab = {}
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
		if self.vocab[content[1]] == nil then self.vocab[content[1]] = 0; end
		if self.vocab[content[2]] == nil then self.vocab[content[2]] = 0; end
		self.vocab[content[1]] = self.vocab[content[1]] + 1
		self.vocab[content[2]] = self.vocab[content[2]] + 1
	end
	print('#nodes = '..#self.id_2_node)
	print('#edges = '..self.edge_count)
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
	self:build_table()
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

-- Build a table of unigram frequencies from which to obtain negative samples
function Node2Vec:build_table()
	local start = sys.clock()
    local total_count_pow = 0
    print("Building a table of unigram frequencies... ")
    for _, count in pairs(self.vocab) do
    	total_count_pow = total_count_pow + count^0.75
    end   
    self.table_size = 100000
    self.table = torch.IntTensor(self.table_size)
    local node_index = 1
    local node_prob = self.vocab[self.id_2_node[node_index]]^0.75 / total_count_pow
    for idx = 1, self.table_size do
        self.table[idx] = node_index
        if idx / self.table_size > node_prob then
            node_index = node_index + 1
	    	node_prob = node_prob + self.vocab[self.id_2_node[node_index]]^0.75 / total_count_pow
        end
        if node_index > #self.id_2_node then
            node_index = node_index - 1
        end
    end
    print(string.format("Done in %.2f seconds.", sys.clock() - start))
end