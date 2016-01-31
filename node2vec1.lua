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
	self.db = lmdb.env{Path = './b111_n2vDB', Name = 'b111_n2vDB'}

	-- create the vocabulary
	self:create_vocab()

	-- create the model
	self:create_model()

	-- create the batches
	self:create_batches()
end

function Node2Vec:create_batches()
	print('creating batches...')
	local start = sys.clock()
	self.db:open()
	local writer = self.db:txn()
	if self.use_weight == 1 then
		local sample_idx = 0
		print('creating batches for m1...')
		local num_batches = torch.floor(self.edge_count / self.m1_batch_size)
		local tensor_a = torch.IntTensor(1)
		local tensor_b = torch.IntTensor(1)
		-- if self.gpu == 1 then tensor_a = tensor_a:cuda(); tensor_b = tensor_b:cuda() end
		xlua.progress(1, num_batches)
		c = 0
		local input, label = {}, torch.Tensor(self.m1_batch_size)		
		-- if self.gpu == 1 then label = label:cuda() end		
		for line in io.lines(self.network_file) do
			line = string.gsub(line, '\n', '')
			local content = string.split(line, '\t')
			local weight = nil
			if #content > 2 then
				weight = tonumber(content[3]) / self.total_weight
			else
				weight = 1 / self.total_weight
			end
			tensor_a[1] = self.node_2_id[content[1]]
			tensor_b[1] = self.node_2_id[content[2]]
			table.insert(input, {tensor_a:cuda(), tensor_b:cuda()})
			label[#input] = weight
			if #input == self.m1_batch_size then
				sample_idx = sample_idx + 1
				writer:put('m1_'..sample_idx, {input, label:cuda()})
				if sample_idx % 50000 == 0 then
					writer:commit()
					writer = self.db:txn()
					collectgarbage()
				end
				input = nil
				input = {}
				if sample_idx % 200 == 0 then
					xlua.progress(sample_idx, num_batches)
				end
			end			
			c = c + 1
			if c == 10000 then
				-- break
			end
		end
		if #input ~= 0 then
			sample_idx = sample_idx + 1
			local cur_size = #input
			for j = cur_size + 1, self.m1_batch_size do
				table.insert(input, input[1])
				label[#input] = label[1]
			end
			writer:put('m1_'..sample_idx, {input, label:cuda()})
			input = nil
			input = {}
			label = nil
		end
		collectgarbage()
		xlua.progress(num_batches, num_batches)
		print(sample_idx..' batches created.')
		writer:put('m1_size', sample_idx)
	end
	if self.use_link == 1 then
		local sample_idx = 0
		print('creating batches for m2...')
		local num_batches = torch.floor(self.edge_count / self.m2_batch_size)
		c = 0
		if self.use_hsm == 0 then
			xlua.progress(1, num_batches)
			local cur_idx = 0
			local input, label = torch.IntTensor(self.m2_batch_size), torch.IntTensor(self.m2_batch_size)
			--[[
			if self.gpu == 1 then
				input = input:cuda()
				label = label:cuda()
			end
			]]--
			for line in io.lines(self.network_file) do
				line = string.gsub(line, '\n', '')
				local content = string.split(line, '\t')
				cur_idx = cur_idx + 1
				input[cur_idx] = self.node_2_id[content[1]]
				label[cur_idx] = self.node_2_id[content[2]]
				if cur_idx == self.m2_batch_size then
					sample_idx = sample_idx + 1
					writer:put('m2_'..sample_idx, {input:cuda(), label:cuda()})
					if sample_idx % 50000 == 0 then
						writer:commit()
						writer = self.db:txn()
						collectgarbage()
					end
					if sample_idx % 200 == 0 then
						xlua.progress(sample_idx, num_batches)
					end
					cur_idx = 0
				end						
				c = c + 1
				if c == 10000 then
					-- break
				end
			end
			if cur_idx ~= 0 then				
				sample_idx = sample_idx + 1
				writer:put('m2_'..sample_idx, {input:cuda(), label:cuda()})
			end
			xlua.progress(num_batches, num_batches)
			print(sample_idx..' batches created.')
			writer:put('m2_size', sample_idx)
		else
			xlua.progress(1, num_batches)
			local input, label = {}, torch.IntTensor(self.m2_batch_size)
			-- if self.gpu == 1 then label = label:cuda() end
			local tensor_a = torch.IntTensor(1)
			local tensor_b = torch.IntTensor(1)
			for line in io.lines(self.network_file) do
				line = string.gsub(line, '\n', '')
				local content = string.split(line, '\t')
				-- local tensor_a, tensor_b = torch.Tensor{self.node_2_id[content[1]]}, torch.Tensor{self.node_2_id[content[2]]}
				-- if self.gpu == 1 then tensor_a = tensor_a:cuda(); tensor_b = tensor_b:cuda() end
				tensor_a[1] = self.node_2_id[content[1]]
				tensor_b[1] = self.node_2_id[content[2]]
				table.insert(input, {tensor_a:cuda(), tensor_b:cuda()})
				label[#input] = self.node_2_id[content[2]]
				if #input == self.m2_batch_size then
					sample_idx = sample_idx + 1
					writer:put('m2_'..sample_idx, {input, label:cuda()})
					if sample_idx % 50000 == 0 then
						writer:commit()
						writer = self.db:txn()
						collectgarbage()
					end
					if sample_idx % 200 == 0 then
						xlua.progress(sample_idx, num_batches)
					end
					input = nil
					input = {}
				end						
				c = c + 1
				if c == 10000 then
					-- break
				end
			end
			if #input ~= 0 then				
				sample_idx = sample_idx + 1
				local cur_size = #input
				for i = cur_size + 1, self.m2_batch_size do
					table.insert(input, input[1])
					label[#input] = label[1]
				end
				writer:put('m2_'..sample_idx, {input, label:cuda()})
			end
			xlua.progress(num_batches, num_batches)
			print(sample_idx..' batches created.')
			writer:put('m2_size', sample_idx)
		end
	end
	if self.use_label == 1 then
		local sample_idx = 0
		print('creating batches for m3...')
		local num_batches = torch.floor(self.labelled_nodes_count / self.m3_batch_size)
		xlua.progress(1, num_batches)
		c = 0
		local cur_idx = 0
		local input, label = torch.Tensor(self.m3_batch_size), nil
		if self.use_multi_label == 1 then
			label = torch.zeros(self.m3_batch_size, #self.id_2_label)
		else
			label = torch.Tensor(self.m3_batch_size)
		end
		-- if self.gpu == 1 then input = input:cuda(); label = label:cuda(); end
		for line in io.lines(self.labels_file) do
			line = string.gsub(line, '\n', '')
			local content = string.split(line, '\t')
			local labs = string.split(content[2], ',')
			if self.use_multi_label == 1 then
				local labels = {}
				for i = 1, #labs do
					table.insert(labels, self.label_2_id[labs[i]])
				end
				table.sort(labels)
				cur_idx = cur_idx + 1
				for i, lab in ipairs(labels) do
					label[cur_idx][i] = lab
				end
				input[cur_idx] = self.node_2_id[content[1]]
				if cur_idx == self.m3_batch_size then
					sample_idx = sample_idx + 1
					writer:put('m3_'..sample_idx, {input:cuda(), label:cuda()})
					if sample_idx % 50000 == 0 then
						writer:commit()
						writer = self.db:txn()
						collectgarbage()
					end
					if sample_idx % 200 == 0 then
						xlua.progress(sample_idx, num_batches)
					end
					cur_idx = 0
				end
			else
				for i = 1, #labs do
					cur_idx = cur_idx + 1
					input[cur_idx] = self.node_2_id[content[1]]
					label[cur_idx] = self.label_2_id[labs[i]]
					if cur_idx == self.m3_batch_size then
						sample_idx = sample_idx + 1
						writer:put('m3_'..sample_idx, {input:cuda(), label:cuda()})
						if sample_idx % 50000 == 0 then
							writer:commit()
							writer = self.db:txn()
							collectgarbage()
						end
						if sample_idx % 200 == 0 then
							xlua.progress(sample_idx, num_batches)
						end
						cur_idx = 0
					end	
				end
			end		
			c = c + 1
			if c == 10000 then
				-- break
			end
		end		
		if cur_idx ~= 0 then
			sample_idx = sample_idx + 1
			writer:put('m3_'..sample_idx, {input:cuda(), label:cuda()})		
		end
		xlua.progress(num_batches, num_batches)
		print(sample_idx..' batches created.')
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
	for epoch = 1, self.max_epochs do
		local epoch_start = sys.clock()

		if self.use_weight == 1 then
			local function threadedTrain(model, criterion, num_threads, reader, is_gpu, batch_size, lr)
				Threads.serialization('threads.sharedserialize')
				local threads = Threads(
					num_threads,
					function()
						require 'nn'
						require 'cunn'
						require 'cutorch'
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
				xlua.progress(1, m1_size)
				for i = 1, m1_size do
					local data = reader:get('m1_'..indices[i])
					local input, label = data[1], data[2]
					threads:addjob(
						function(input, label)
							return gupdate(input, label)
						end,
						function(err, dweights)
							epoch_loss = epoch_loss + err
							epoch_iteration = epoch_iteration + 1
							for i = 1, #weights do
								weights[i]:add(-1 * lr, dweights[i])
							end
							if epoch_iteration % 5 == 0 then
								collectgarbage()
								xlua.progress(epoch_iteration, m1_size)
							end
						end,
						input,
						label
					)
				end
				threads:synchronize()
				threads:terminate()
				xlua.progress(m1_size, m1_size)
				print(string.format("Done in %.2f minutes. Weight-Info Loss = %f\n",((sys.clock() - m1_start) / 60), (epoch_loss / epoch_iteration)))
			end
			threadedTrain(self.weight_model, self.weight_criterion, self.num_threads, reader, self.gpu, self.m1_batch_size, self.learning_rate)
		end

		if self.use_link == 1 then
			local function threadedTrain(model, criterion, num_threads, reader, is_gpu, batch_size, lr, use_hsm)
				Threads.serialization('threads.sharedserialize')
				local threads = Threads(
					num_threads,
					function()
						require 'nn'
						require 'nnx'
						require 'cunn'
						require 'cunnx'
						require 'cutorch'
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
								dweights[wi] = dweights[wi]:div(batch_size)
							end
							loss = loss / batch_size
							return loss, dweights
						end
					end
				)
				local weights = model:parameters()
				local m2_size = reader:get('m2_size')
				local indices = torch.randperm(m2_size)
				local epoch_loss, epoch_iteration = 0, 0
				local m2_start = sys.clock()
				xlua.progress(1, m2_size)
				for i = 1, m2_size do
					local data = reader:get('m2_'..indices[i])
					input, label = data[1], data[2]
					threads:addjob(
						function(input, label)
							return gupdate(input, label)
						end,
						function(err, dweights)
							epoch_loss = epoch_loss + err
							epoch_iteration = epoch_iteration + 1
							for i = 1, #weights do
								weights[i]:add(-1 * lr, dweights[i])
							end
							if epoch_iteration % 5 == 0 then
								collectgarbage()
								xlua.progress(epoch_iteration, m2_size)
							end
						end,
						input,
						label
					)			
				end
				threads:synchronize()
				threads:terminate()
				xlua.progress(m2_size, m2_size)
				print(string.format("Done in %.2f minutes. Link-Info Loss = %f\n",((sys.clock() - m2_start) / 60), (epoch_loss / epoch_iteration)))
			end
			threadedTrain(self.link_model, self.link_criterion, self.num_threads, reader, self.gpu, self.m2_batch_size, self.learning_rate, self.use_hsm)
		end

		if self.use_label == 1 then
			local function threadedTrain(model, criterion, num_threads, reader, is_gpu, batch_size, lr, use_reg, reg)
				Threads.serialization('threads.sharedserialize')
				local threads = Threads(
					num_threads,
					function()
						require 'nn'
						require 'nnx'
						require 'cunn'
						require 'cutorch'
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
								dweights[wi] = dweights[wi]:div((#input)[1])
							end
							loss = loss / (#input)[1]
							return loss, dweights
						end
					end
				)
				local weights = model:parameters()
				local m3_size = reader:get('m3_size')
				local indices = torch.randperm(m3_size)
				local epoch_loss, epoch_iteration = 0, 0
				local m3_start = sys.clock()
				xlua.progress(1, m3_size)
				for i = 1, m3_size do
					local data = reader:get('m3_'..indices[i])
					local input, label = data[1], data[2]
					threads:addjob(
						function(input, label)
							return gupdate(input, label)
						end,
						function(err, dweights)
							epoch_loss = epoch_loss + err
							epoch_iteration = epoch_iteration + 1
							for i = 1, #weights do
								weights[i]:add(-1 * lr, dweights[i])
							end
							if epoch_iteration % 5 == 0 then
								collectgarbage()
								xlua.progress(epoch_iteration, m3_size)
							end
						end,
						input,
						label
					)
				end
				threads:synchronize()
				threads:terminate()
				xlua.progress(m3_size, m3_size)
				print(string.format("Done in %.2f minutes. Label-Info Loss = %f\n",((sys.clock() - m3_start) / 60), (epoch_loss / epoch_iteration)))
			end
			threadedTrain(self.label_model, self.label_criterion, self.num_threads, reader, self.gpu, self.m3_batch_size, self.learning_rate)
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
			xlua.progress(1, m1_size)
			self.m1_optim_state.learningRate = self.learning_rate
			for i = 1, m1_size do
				local data = reader:get('m1_'..indices[i])
				self.m1_input, self.m1_label = data[1], data[2]
				local _, loss = optim.sgd(self.m1_feval, self.m1_params, self.m1_optim_state)
				epoch_loss = epoch_loss + loss[1]
				epoch_iteration = epoch_iteration + 1
				if epoch_iteration % 5 == 0 then
					xlua.progress(epoch_iteration, m1_size)
					collectgarbage()
				end
				self.m1_optim_state.learningRate = self.m1_optim_state.learningRate * (1 - (i / m1_size))
			end
			xlua.progress(m1_size, m1_size)
			print(string.format("Done in %.2f minutes. Weight-Info Loss = %f\n",((sys.clock() - m1_start) / 60), (epoch_loss / epoch_iteration)))
		end
		if self.use_link == 1 then
			local m2_size = reader:get('m2_size')
			local indices = torch.randperm(m2_size)
			local m2_start = sys.clock()
			local epoch_loss, epoch_iteration = 0, 0
			xlua.progress(1, m2_size)
			self.m2_optim_state.learningRate = self.learning_rate
			for i = 1, m2_size do
				local data = reader:get('m2_'..indices[i])
				self.m2_input, self.m2_label = data[1], data[2]
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
		if self.use_label == 1 then
			local m3_size = reader:get('m3_size')
			local indices = torch.randperm(m3_size)
			local m3_start = sys.clock()
			local epoch_loss, epoch_iteration = 0, 0
			xlua.progress(1, m3_size)
			self.m3_optim_state.learningRate = self.learning_rate
			for i = 1, m3_size do
				local data = reader:get('m3_'..indices[i])
				self.m3_input, self.m3_label = data[1], data[2]
				local _, loss = optim.sgd(self.m3_feval, self.m3_params, self.m3_optim_state)
				epoch_loss = epoch_loss + loss[1]
				epoch_iteration = epoch_iteration + 1
				if epoch_iteration % 5 == 0 then
					xlua.progress(epoch_iteration, m3_size)
					collectgarbage()
				end
				self.m3_optim_state.learningRate = self.m3_optim_state.learningRate * (1 - (i / m3_size))
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
		print('creating weight model...')
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
		xlua.progress(1, self.m1_batch_size)		
		for i = 1, self.m1_batch_size do
			self.weight_model.modules[1]:add(self._weight_model:clone('weight', 'bias', 'gradWeight', 'gradBias'))
			if i % 10 == 0 then
				xlua.progress(i, self.m1_batch_size)
			end
		end
		xlua.progress(self.m1_batch_size, self.m1_batch_size)
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
			local pred = self.weight_model:forward(self.m1_input)
			local loss = self.weight_criterion:forward(pred, self.m1_label)
			local grads = self.weight_criterion:backward(pred, self.m1_label)
			self.weight_model:backward(self.m1_input, grads)
			self.m1_grad_params:div(#self.m1_input)			
			loss = loss / #self.m1_input
			if self.use_reg == 1 then
				loss = loss + 0.5 * self.reg * self.m1_params:norm()^2
			end
			return loss, self.m1_grad_params
		end
	end

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
		self.link_model = nil
		if self.use_hsm == 1 then
			self.tree, self.root = self:create_frequency_tree(#self.id_2_node)
			self.__link_model = nn.Sequential()
			self.__link_model:add(nn.ParallelTable())
			self.__link_model.modules[1]:add(self._link_model)
			self.__link_model.modules[1]:add(nn.Identity())
			self.__link_model:add(nn.SoftMaxTree(input_size, self.tree, self.root))
			self.link_criterion = nn.TreeNLLCriterion()
			self.link_model = nn.Sequential()
			self.link_model:add(nn.ParallelTable())
			xlua.progress(1, self.m2_batch_size)
			for i = 1, self.m2_batch_size do
				self.link_model.modules[1]:add(self.__link_model:clone('weight', 'bias', 'gradWeight', 'gradBias'))
				if i % 10 == 0 then
					xlua.progress(i, self.m2_batch_size)
				end
			end
			xlua.progress(self.m2_batch_size, self.m2_batch_size)
			self.link_model:add(nn.JoinTable(1))
		else
			self.link_model = self._link_model
			self.link_model:add(nn.Linear(input_size, #self.id_2_node))
			self.link_criterion = nn.CrossEntropyCriterion()
		end
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

	if self.use_label == 1 then
		print('creating label model...')
		self.label_model = nn.Sequential()
		self.label_model:add(self.node_vecs:clone('weight', 'bias', 'gradWeight', 'gradBias'))
		local layers = string.split(self.m3_layers, ',')
		local input_size = self.embed_size
		for i = 1, #layers do
			self.label_model:add(nn.Linear(input_size, tonumber(layers[i])))
			input_size = tonumber(layers[i])
		end
		self.label_model:add(nn.Linear(input_size, #self.id_2_label))		
		self.label_criterion = nil
		if self.use_multi_label == 0 then self.label_criterion = nn.CrossEntropyCriterion(); else self.label_criterion = nn.MultiLabelMarginCriterion(); end
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
			local pred = self.label_model:forward(self.m3_input)
			local loss = self.label_criterion:forward(pred, self.m3_label)
			local grads = self.label_criterion:backward(pred, self.m3_label)
			self.label_model:backward(self.m3_input, grads)
			self.m3_grad_params:div((#self.m3_input)[1])
			loss = loss / ((#self.m3_input)[1])
			if self.use_reg == 1 then
				loss = loss + 0.5 * self.reg * self.m3_params:norm()^2
			end
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
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

function Node2Vec:create_frequency_tree(node_count)
	freq_map = self:create_node_map(node_count)
	binSize = self.hsm_bins
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