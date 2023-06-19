import torch
import os
from network.base_net import RNN
import numpy as np


class ICL:
	def __init__(self, args):
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape
		input_shape = self.obs_shape

		# input dimension for rnn according to the params
		if args.last_action:
		    input_shape += self.n_actions
		
		self.eval_rnn = RNN(input_shape, args)  # each agent picks a net of actions
		self.target_rnn = RNN(input_shape, args)

		self.args = args

		# cuda
		if self.args.cuda:
			self.eval_rnn.cuda()
			self.target_rnn.cuda()

		self.model_dir = args.model_dir + '/' + args.alg

		if self.args.load_model:
		    if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
		        path_rnn = self.model_dir + '/rnn_net_params.pkl'
		        path_vdn = self.model_dir + '/vdn_net_params.pkl'
		        print('Successfully load the model: {} and {}'.format(path_rnn, path_vdn))
		    else:
		    	raise Exception("No such model!")

		# make parameters of target and eval the same
		self.target_rnn.load_state_dict(self.eval_rnn.state_dict())

		self.eval_parameters = list(self.eval_rnn.parameters()) 
		if args.optimizer == "RMS":
		    self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)


		# during learning one should keep an eval_hidden and a target_hidden for each agent of each episode
		self.eval_hidden = None
		self.target_hidden = None

		print("ICL algorithm initialized")


	def learn(self, batch, max_episode_len, train_step, agent_id, epsilon=None):  
		""" Method where the causality factors for the environments are calculated """
		raise NotImplementedError

		
	def get_q_values(self, batch, max_episode_len, agent_id):
		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		q_evals, q_targets = [], []
		for transition_idx in range(max_episode_len):
		    inputs, inputs_next = self._get_inputs(batch, transition_idx, agent_id)  # add last action and agent_id to the obs

		    # cuda
		    if self.args.cuda:
		    	inputs = inputs.cuda()
		    	inputs_next = inputs_next.cuda()
		    	self.eval_hidden = self.eval_hidden.cuda()
		    	self.target_hidden = self.target_hidden.cuda()
		    
		    q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  
		    q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

		    # Change the q_eval dimension back to (8, 5(n_agents), n_actions)
		    q_eval = q_eval.view(episode_num, -1)
		    q_target = q_target.view(episode_num, -1)
		    q_evals.append(q_eval)
		    q_targets.append(q_target)

		'''
		q_eval and q_target are lists containing max_episode_len arrays with dimensions (episode_number, n_agents, n_actions)
		convert the lists into arrays of (episode_number, max_episode_len, n_agents, n_actions)
		'''

		q_evals = torch.stack(q_evals, dim=1)
		q_targets = torch.stack(q_targets, dim=1)
		return q_evals, q_targets

	def _get_inputs(self, batch, transition_idx, agent_id):
		obs, obs_next, actions_onehot = batch['obs'][:, transition_idx, agent_id], \
		                          batch['obs_next'][:, transition_idx, agent_id], batch['actions_onehot'][:]
		episode_num = obs.shape[0]
		inputs, inputs_next = [], []
		inputs.append(obs)
		inputs_next.append(obs_next)

		# adds last action and agent number to obs
		if self.args.last_action:
		    if transition_idx == 0:  # if it is the first transition, let the previous action be a 0 vector
		        inputs.append(torch.zeros_like(actions_onehot[:, transition_idx, agent_id]))
		    else:
		        inputs.append(actions_onehot[:, transition_idx - 1, agent_id])
		    inputs_next.append(actions_onehot[:, transition_idx, agent_id])

		inputs = torch.cat([x.reshape(episode_num, -1) for x in inputs], dim=1)
		inputs_next = torch.cat([x.reshape(episode_num, -1) for x in inputs_next], dim=1)

		return inputs, inputs_next



	def init_hidden(self, episode_num):
		# initializes eval_hidden and target_hidden for each agent of each episode, as in DQN there is a net and a target net to stabilize learning

		self.eval_hidden = torch.zeros((episode_num, self.args.rnn_hidden_dim))
		self.target_hidden = torch.zeros((episode_num, self.args.rnn_hidden_dim))


	def save_model(self, train_step, agent_id, end_training=False):
		# save final policies at the end of training
		if end_training:
			torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + f'final_rnn_net_params_{agent_id}.pkl')
		else:
			num = str(train_step // self.args.save_cycle)
			if not os.path.exists(self.model_dir):
				os.makedirs(self.model_dir)
			torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + f'_rnn_net_params_{agent_id}.pkl')

