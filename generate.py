from LINKS import *
import argparse

parser = argparse.ArgumentParser(description='DataGen Parameters')

parser.add_argument('--N', type=int, default=10000, help='Number of Topologies')
parser.add_argument('--save_name', type=str, default='dataset', help='Save name will save as mechanisms as "save_name". If already exisitin will append')

args = parser.parse_args()

print('Generating Topologies ...')
topologies = run_imap_multiprocessing(get_random_topology,[None]*args.N)

print('Generating Initial Positions ...')
candidates = run_imap_multiprocessing(get_candidates_for_topology,topologies)


mechanisms = []

for i in range(len(candidates)):
    A,motor,fixed_nodes = topologies[i]
    for j in range(5):
        x0 = candidates[i][j].astype(np.float32)
        node_type = np.zeros([A.shape[0]]).astype(np.bool)
        node_type[fixed_nodes] = 1
        mechanisms.append([A.astype(np.bool), x0, node_type])

print('Processin Data ...')

sols = run_imap_multiprocessing(simulate_mechanism,mechanisms)

xsol = []
xsoln = []
cur = []

for j,s in enumerate(sols):
    if not s is None:
        x_sol, x_sol_n, moving_nodes, x_norm_cur, x_norm_cur_i = s
        xsol.append([x_sol,j])
        
        for k in range(moving_nodes.shape[0]):
            xsoln.append([x_sol_n[k],j,moving_nodes[k]])
            
        for k in range(x_norm_cur_i.shape[0]):
            cur.append([x_norm_cur[k],j,x_norm_cur_i[k]])

if os.path.exists(args.save_name):
    with open(args.save_name, 'ab') as f:
        pickle.dump(mechanisms,f)
else:
    with open(args.save_name, 'wb') as f:
        pickle.dump(mechanisms,f)

if os.path.exists('Solutions_' + args.save_name):
    with open('Solutions_' + args.save_name, 'ab') as f:
        pickle.dump(xsol,f)
else:
    with open('Solutions_' + args.save_name, 'wb') as f:
        pickle.dump(xsol,f)

if os.path.exists('Normalized_' + args.save_name):
    with open('Normalized_' + args.save_name, 'ab') as f:
        pickle.dump(xsoln,f)
else:
    with open('Normalized_' + args.save_name, 'wb') as f:
        pickle.dump(xsoln,f)

if os.path.exists('Curated_Normalized_' + args.save_name):
    with open('Curated_Normalized_' + args.save_name, 'ab') as f:
        pickle.dump(cur,f)
else:
    with open('Curated_Normalized_' + args.save_name, 'wb') as f:
        pickle.dump(cur,f)