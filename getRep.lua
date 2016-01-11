obj = torch.load('node_3.t7')
weight = obj.node_weights
node_2_id = obj.node_2_id
id_2_node = obj.id_2_node
writer = io.open('rep', 'w')
for i = 1, (#weight)[1] do
	writer:write(id_2_node[i])
	for j = 1, (#weight)[2] do
		writer:write('\t'..weight[i][j])
	end
	writer:write('\n')
end
writer.close()