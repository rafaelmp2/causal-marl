from argparse import ArgumentParser
import torch

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
    parser.add_argument('--save_cycle', type=int, default=3333, help='how often to save the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')

    # args for the acd framework

    parser.add_argument("--global_temp",action="store_true",default=False,help="Should we model temperature confounding?")
    parser.add_argument("--dims", type=int, default=4, help="Dimensionality of input.")

    parser.add_argument(
    "--encoder_hidden", type=int, default=256, help="Number of hidden units."
    )
    parser.add_argument(
    "--decoder_hidden", type=int, default=256, help="Number of hidden units."
    )
    parser.add_argument(
    "--encoder",
    type=str,
    default="mlp",
    help="Type of path encoder model (mlp or cnn).",
    )
    parser.add_argument(
    "--decoder",
    type=str,
    default="mlp",
    help="Type of decoder model (mlp, rnn, or sim).",
    )
    parser.add_argument(
    "--prior",
    type=float,
    default=1,
    help="Weight for sparsity prior (if == 1, uniform prior is applied)",
    )
    parser.add_argument(
    "--edge_types",
    type=int,
    default=2,
    help="Number of different edge-types to model",
    )

    parser.add_argument(
    "--encoder_dropout",
    type=float,
    default=0.0,
    help="Dropout rate (1 - keep probability).",
    )
    parser.add_argument(
    "--decoder_dropout",
    type=float,
    default=0.0,
    help="Dropout rate (1 - keep probability).",
    )

    parser.add_argument(
    "--no_factor",
    action="store_true",
    default=False,
    help="Disables factor graph model.",
    )

    ### unobserved time-series ###
    parser.add_argument(
    "--unobserved",
    type=int,
    default=0,
    help="Number of time-series to mask from input.",
    )
    parser.add_argument(
    "--model_unobserved",
    type=int,
    default=0,
    help="If 0, use NRI to infer unobserved particle. "
    "If 1, removes unobserved from data. "
    "If 2, fills empty slot with mean of observed time-series (mean imputation)",
    )
    parser.add_argument(
    "--dont_shuffle_unobserved",
    action="store_true",
    default=False,
    help="If true, always mask out last particle in trajectory. "
    "If false, mask random particle.",
    )
    parser.add_argument(
    "--teacher_forcing",
    type=int,
    default=0,
    help="Factor to determine how much true trajectory of "
    "unobserved particle should be used to learn prediction.",
    )
    parser.add_argument(
    "--load_folder",
    type=str,
    default="",
    help="Where to load pre-trained model if finetuning/evaluating. "
    + "Leave empty to train from scratch",
    )

    parser.add_argument(
    "--GPU_to_use", type=int, default=None, help="GPU to use for training"
    )

    parser.add_argument(
        "--dont_skip_first",
        action="store_true",
        default=False,
        help="If given as argument, do not skip first edge type in decoder, i.e. it represents no-edge.",
    )

    parser.add_argument(
        "--dont_use_encoder",
        action="store_true",
        default=False,
        help="If true, replace encoder with distribution to be estimated",
    )

    parser.add_argument(
        "--lr_decay",
        type=int,
        default=200,
        help="After how epochs to decay LR by a factor of gamma.",
    )
    
    parser.add_argument(
        "--temp", type=float, default=0.5, help="Temperature for Gumbel softmax."
    )

    args = parser.parse_args()

    args.num_GPU = 0
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.cuda:
        args.num_GPU = 1  

    args.factor = not args.no_factor
    args.skip_first = not args.dont_skip_first
    args.use_encoder = not args.dont_use_encoder

    # in accordance to acd framework
    if "PredatorPrey" in args.env:
        args.timesteps = 100
        args.dims = 25 
        args.num_atoms = 6

    if "Lumberjacks" in args.env:
        args.timesteps = 100 
        args.dims = 18 
        args.num_atoms = 5

    if "3m" in args.env:
        args.timesteps = 60 
        args.dims = 1 
        args.num_atoms = 4

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

