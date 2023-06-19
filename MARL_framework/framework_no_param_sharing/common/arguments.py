from argparse import ArgumentParser

def common_args():
	parser = ArgumentParser()
	
	# smac args
	parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
	parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
	parser.add_argument('--map', type=str, default='3m', help='the map of the game')
	parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
	parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')

	# general args
	parser.add_argument("--env", "-e", default="Switch2-v0", help="set env name")
	parser.add_argument("--n_steps", "-ns", type=int, default=2000000, help="set total time steps to run")
	parser.add_argument("--n_episodes", "-nep", type=int, default=1, help="set n_episodes")
	parser.add_argument("--epsilon", "-eps", default=0.5, help="set epsilon value")
	parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
	parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
	parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
	parser.add_argument('--evaluate_epoch', type=int, default=20, help='the number of the epoch to evaluate the agent')
	parser.add_argument('--alg', type=str, default='vdn', help='the algorithm to train the agent')
	parser.add_argument('--optimizer', type=str, default="RMS", help='the optimizer')
	parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
	parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
	parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
	parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
	parser.add_argument('--evaluate_cycle', type=int, default=5000, help='how often to eval the model')
	parser.add_argument('--target_update_cycle', type=int, default=200, help='how often to update the target network')
	parser.add_argument('--save_cycle', type=int, default=6650, help='how often to save the model')
	parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')

	args = parser.parse_args()

	return args


def config_args(args):
	# buffer/batch sizes
	args.batch_size = 32
	args.buffer_size = int(5e3)
	
	#epsilon args for vdn
	args.epsilon = 1
	args.min_epsilon = 0.05
	anneal_steps = 50000
	args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
	args.epsilon_anneal_scale = 'step'

	# network
	args.rnn_hidden_dim = 64
	args.lr = 5e-4

	# train steps
	args.train_steps = 1

	# prevent gradient explosion
	args.grad_norm_clip = 10	

	return args

