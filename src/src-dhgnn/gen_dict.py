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


print("hg shape:", hg.shape)
print("hg: \n", hg)
con_e = {}
adj_u = {}

hyper_edges_id = sorted(list(set(hg[1].numpy().tolist())))

# np.where(values == searchval)[0]
a = 0
for ele in set(hyper_edges_id):
	ids = np.where(hg[1].numpy() == ele)[0]
	# print(hg[0][ids])
	con_e[ele] = hg[0][ids].numpy().tolist()
	a += len(hg[0][ids].numpy().tolist())
print("__"*80)
print("con_e:\n", con_e)
# print("__"*80)
# print(a, len(con_e))
print("__"*80)


vertices_id = sorted(list(set(hg[0].numpy().tolist())))
b = 0
# np.where(values == searchval)[0]
for ele in set(vertices_id):
	ids = np.where(hg[0].numpy() == ele)[0]
	# print(hg[0][ids])
	adj_u[ele] = hg[1][ids].numpy().tolist()
	b += len(hg[1][ids].numpy().tolist())

print("adj_u:\n", adj_u)
# print("__"*80)
# print(b, len(adj_u))
print("__"*80)

doc_emb = torch.randn(116, 768)
price_emb = torch.randn(116, 1)

# print(price_emb[[16, 17]])
print("__"*80)
print("__"*80)
for vertex, hyper_edge_set in adj_u.items():
	hyper_edge_list = []
	print(vertex, hyper_edge_set)
	for hyper_edge in hyper_edge_set:
		a = price_emb[con_e[hyper_edge]]
		b = doc_emb[hyper_edge]
		# print(a)
		print(a.shape, b.shape)
	break