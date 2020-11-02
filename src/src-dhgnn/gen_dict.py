import config
import os 
import pickle 
import numpy as np 
from scipy import sparse
from torch_geometric import utils
print("__"*80)
print("Imports done...")
print("__"*80)

dates = sorted(open(config.DATES_PATH, "r").read().split())
print(dates[0], len(dates))

for idx, date in enumerate(dates):
	hg_ = np.load(config.HG_PATH + date + ".npy")
	num_edges = hg_.shape[1]
	print(f"{idx+1} Date: {date} num_edges: {hg_.shape[1]}")
	#* Process the npy hg to feed it to the hgnn
	inci_sparse = sparse.coo_matrix(hg_)
	incidence_edge = utils.from_scipy_sparse_matrix(inci_sparse)
	hyp_input = incidence_edge[0] # this is edge list (2, x)

	hg = hyp_input
	con_e = {}
	adj_u = {}
	hyper_edges_id = sorted(list(set(hg[1].numpy().tolist())))
	for ele in set(hyper_edges_id):
		ids = np.where(hg[1].numpy() == ele)[0]
		con_e[ele] = hg[0][ids].numpy().tolist()

	vertices_id = sorted(list(set(hg[0].numpy().tolist())))
	for ele in set(vertices_id):
		ids = np.where(hg[0].numpy() == ele)[0]
		adj_u[ele] = hg[1][ids].numpy().tolist()

	# print("__"*80)
	# print("hg:")
	# print(hg)
	# print("__"*80)
	# print("con_e:")
	# print(con_e)
	# print("__"*80)
	# print("adj_u:")
	# print(adj_u)
	# print("__"*80)
	try: 
		if not os.path.exists(config.CON_E_PATH):
			os.mkdir(config.CON_E_PATH)
		con_e_file = open(os.path.join(config.CON_E_PATH, date), 'wb') 
		pickle.dump(con_e, con_e_file) 
		con_e_file.close()
		
	except: 
		print(f"Something went wrong in con_e of {date}")

	try: 
		if not os.path.exists(config.ADJ_U_PATH):
			os.mkdir(config.ADJ_U_PATH)
		adj_u_file = open(os.path.join(config.ADJ_U_PATH, date), 'wb') 
		pickle.dump(adj_u, adj_u_file) 
		adj_u_file.close() 
		
	except: 
		print(f"Something went wrong in adj_u of {date}")
	# break

  
print("__"*80)
print("__"*80)
print("__"*80)
print("loaded:")
print(date)
print()
infile = open(os.path.join(config.CON_E_PATH, date),'rb')
con_e = pickle.load(infile)
print(con_e)
infile.close()

infile = open(os.path.join(config.ADJ_U_PATH, date),'rb')
adj_u = pickle.load(infile)
infile.close()
print(adj_u)