cmd = torch.CmdLine()
cmd:option('-src','','')
cmd:option('-dest','','')
params = cmd:parse(arg)

obj = torch.load(params.src)
weight = obj.node_weights
node_2_id = obj.node_2_id
id_2_node = obj.id_2_node
writer = io.open(params.dest, 'w')
for i = 1, (#weight)[1] do
	writer:write(id_2_node[i])
	for j = 1, (#weight)[2] do
		writer:write('\t'..weight[i][j])
	end
	writer:write('\n')
end
writer.close()