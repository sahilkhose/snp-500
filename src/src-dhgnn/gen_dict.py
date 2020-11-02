import torch 
import numpy as np 

hg = torch.tensor([
[ 22,  27,  45, 113],
[  0,   0,   1,   0]
])

# hg = torch.tensor([
# [7,   8,   8,   8,   9,  16,  17,  17,  21,  37,  37,  45,  45,  48, 55,  55,  68,  68,  73,  79,  85,  94,  94,  96, 108, 112, 112, 112, 113, 113],
# [26,   9,  16,  17,  32,  17,   9,  26,  17,   9,  21,   9,  26,   9, 9,  26,  16,  17,  19,  21,  23,   9,  30,   3,  30,   3,  26,  29, 9,  10]
# ])



con_e = {}
adj_u = {}
# hg = hg.squeeze(0)
hyper_edges_id = sorted(list(set(hg[1].numpy().tolist())))
for ele in set(hyper_edges_id):
	ids = np.where(hg[1].numpy() == ele)[0]
	con_e[ele] = hg[0][ids].numpy().tolist()

vertices_id = sorted(list(set(hg[0].numpy().tolist())))
for ele in set(vertices_id):
	ids = np.where(hg[0].numpy() == ele)[0]
	adj_u[ele] = hg[1][ids].numpy().tolist()

print("__"*80)
print("hg:")
print(hg)
print("__"*80)
print("con_e:")
print(con_e)
print("__"*80)
print("adj_u:")
print(adj_u)
print("__"*80)