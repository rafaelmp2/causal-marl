
import gym
import ma_gym
from common.arguments import common_args, config_args
from runner import Runner
from smac.env import StarCraft2Env


#import warnings
#warnings.filterwarnings("ignore", category=UserWarning) 

N_EXPERIMENTS = 1


if __name__ == '__main__':
	args = common_args()	

	'''
	n_actions: number of actions in the environment for each agent
	n_agents: number of agents in the environment
	state_shape: a state is an array with all the values that describe the current state, i.e., all the
	features of the state
	obs_shape: an observation in a partially observable env is what each agent can see; an array with the
	values that describe what each agent can see
	episode_limit: maximum number of steps until which the episode will run if a terminal state wasnt reached
	before

	'''

	if args.env == 'PredatorPrey-v0':
		# avoid registering a new environment in the ma_gym package
		env = gym.make('PredatorPrey7x7-v0', grid_shape=(14, 14), n_agents=4, n_preys=2)
		args.n_actions = env.action_space[0].n
		args.n_agents = env.n_agents
		args.state_shape = 28 * args.n_agents 
		args.obs_shape = 28
		args.episode_limit = env._max_steps
		args.range_dims = [i for i in range(2, args.obs_shape - 1)]
	elif args.env == 'Lumberjacks-v0':
		env = gym.make(args.env, grid_shape=(8,8), n_agents=4)
		args.n_actions = env.action_space[0].n
		args.n_agents = env.n_agents
		args.state_shape = 22 * args.n_agents
		args.obs_shape = 22
		args.episode_limit = env._max_steps
		args.range_dims = [i for i in range(4, args.obs_shape)]
	elif args.env == '3m':
		env = StarCraft2Env(map_name=args.env)
		env_info = env.get_env_info()
		args.n_actions = env_info["n_actions"]
		args.n_agents = env_info["n_agents"]
		args.state_shape = env_info["state_shape"]
		args.obs_shape = env_info["obs_shape"]
		args.episode_limit = env_info["episode_limit"]
		args.range_dims = [4, 9, 14]
	elif args.env == 'Lumberjacks-sp-v0':  # sparser to eval causality
		env = gym.make("Lumberjacks-v0", grid_shape=(5,5), n_agents=4, n_trees=1)  
		args.n_actions = env.action_space[0].n
		args.n_agents = env.n_agents
		args.state_shape = 22 * args.n_agents
		args.obs_shape = 22
		args.episode_limit = env._max_steps
		args.range_dims = [i for i in range(4, args.obs_shape)]
	elif args.env == '3m-sp':  # sparser to eval causality
		env = StarCraft2Env(map_name=args.env)
		env_info = env.get_env_info()
		args.n_actions = env_info["n_actions"]
		args.n_agents = env_info["n_agents"]
		args.state_shape = env_info["state_shape"]
		args.obs_shape = env_info["obs_shape"]
		args.episode_limit = env_info["episode_limit"]
		args.range_dims = [4]
	elif args.env == 'PredatorPrey-sp-v0':  # sparser to eval causality
		env = gym.make('PredatorPrey7x7-v0', grid_shape=(10, 10), n_agents=5, n_preys=1)  
		args.n_actions = env.action_space[0].n
		args.n_agents = env.n_agents
		args.state_shape = 28 * args.n_agents
		args.obs_shape = 28
		args.episode_limit = env._max_steps
		args.range_dims = [i for i in range(2, args.obs_shape - 1)]
	else:
		raise Exception('Invalid environment: environment not supported!')
		

	print("Environment {} initialized, for {} time steps and evaluating every {} time steps".format(args.env, \
																							args.n_steps, args.evaluate_cycle))

	# load args
	# this code is prepared for no parameter sharing methods; so no support for CTDE with parameter sharing methods here
	if args.alg == 'idql' or args.alg == 'icl' or args.alg == 'acd_marl':
		args = config_args(args)
	else:
		raise Exception('No such algorithm!')

	print("CUDA set to", args.cuda)


	runner = Runner(env, args)

	# parameterize run according to the number of independent experiments to run, i.e., independent sets of n_epochs over the model; default is 1
	if args.learn:
		runner.run(N_EXPERIMENTS)
