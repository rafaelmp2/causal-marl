import torch
import os
import numpy as np

from algos.icl import ICL


class ICL_lj(ICL):
	def __init__(self, args):
		super().__init__(args)

	def learn(self, batch, max_episode_len, train_step, agent_id, epsilon=None):  

		'''
			batch: batch with episode batches from before training the model
			max_episode_len: len of the longest episode batch in batch
			train_step: it is used to control and update the params of the target network
			agent_id: id of agent i

			------------------------------------------------------------------------------

			the extracted data is 4D, with meanings 1-> n_episodes, 2-> n_transitions in the episode, 
			3-> data of multiple agents, 4-> obs dimensions
			hidden_state is related to the previous experience so one cant randomly extract
			experience to learn, so multiple episodes are extracted at a time and then given to the
			nn one at a time   
		'''

		episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch
		self.init_hidden(episode_num)

		#convert data in batch to tensor
		for key in batch.keys():  
		    if key == 'actions':
		        batch[key] = torch.tensor(batch[key], dtype=torch.long)
		    else:
		        batch[key] = torch.tensor(batch[key], dtype=torch.float32)

		obs, obs_next, actions, reward, avail_actions, avail_actions_next, terminated = batch['obs'][:, :, agent_id], \
		                          	batch['obs_next'][:, :, agent_id], batch['actions'][:, :, agent_id], batch['reward'], \
		                            batch['avail_actions'][:, :, agent_id], batch['avail_actions_next'][:, :, agent_id], batch['terminated']

		# used to set the td error of the filled experiments to 0, not to affect learning
		mask = 1 - batch["padded"].float()  

		state = batch['state']

		# cuda
		if self.args.cuda:
			obs = obs.cuda()
			obs_next = obs_next.cuda()
			actions = actions.cuda()
			reward = reward.cuda()
			mask = mask.cuda()
			terminated = terminated.cuda()


		# causal heuristics stuff
		obs_aux = obs[:, :, 4:]
		mid = 8
		
		field_of_view = torch.tensor([mid - 6, mid - 5, mid - 2, mid - 1, mid, mid + 1, mid + 2, mid + 3, mid + 6, mid + 7]).repeat(obs.size(0), obs.size(1), 1)
		if self.args.cuda:
			field_of_view = field_of_view.cuda()
		# used to see the preys around
		obs_mask_fov = torch.gather((obs[:, :, 4:]), dim=2, index=field_of_view)

		# selects positions from mask
		obs_mask_fov = obs_mask_fov.reshape(episode_num, obs.shape[1], -1, 2)
		# whole obs mask
		obs_mask_fov_agents = obs_aux.reshape(episode_num, obs.shape[1], -1, 2)

		# saves values regarding agents in the whole obs mask
		obs_mask_agents = torch.index_select(obs_mask_fov_agents, dim=-1, index=torch.tensor([0]).cuda()).squeeze(-1)
		# saves values regarding trees in the selected positions of the mask
		obs_mask_trees = torch.index_select(obs_mask_fov, dim=-1, index=torch.tensor([1]).cuda()).squeeze(-1)

		r_aux = reward.clone()

		# sums values in the agents mask for trees and agents at each time step of each episode
		sum_obs_mask_agents = torch.sum(obs_mask_agents, dim=-1, keepdim=True)
		sum_obs_mask_trees = torch.sum(obs_mask_trees, dim=-1, keepdim=True)

		# cond 1: if agents level in whole mask is >= trees level in capture cells
		cond_aux = torch.where(sum_obs_mask_agents >= sum_obs_mask_trees, 1, 0)

		# cond 2: if there are trees in the capture cells
		sum_aux = torch.where(sum_obs_mask_trees > 0., 1, 0)

		# cond 1 and cond 2 
		reward_condition = sum_aux * cond_aux

		reward_aux = torch.where(reward > 0., reward * reward_condition, reward)

		# gets q value corresponding to each agent, dimensions are (episode_number, max_episode_len, n_agents, n_actions)
		q_evals, q_targets = self.get_q_values(batch, max_episode_len, agent_id)

		# get q value corresponding to each agents action and remove last dim
		q_evals = torch.gather(q_evals, dim=2, index=actions)

		q_targets[avail_actions_next == 0.0] = - 9999999
		q_targets = q_targets.max(dim=2, keepdim=True)[0]

		targets = reward_aux + self.args.gamma * q_targets * (1 - terminated)

		td_error = targets.detach() - q_evals
		masked_td_error = mask * td_error  

		# there are still useless experiments, so the avg is according the number of real experiments
		loss = (masked_td_error ** 2).sum() / mask.sum()

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
		self.optimizer.step()

		# update target networks
		if train_step > 0 and train_step % self.args.target_update_cycle == 0:
		    self.target_rnn.load_state_dict(self.eval_rnn.state_dict())