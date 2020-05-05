import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

sns.set(color_codes=True)
sns.set_style("white")

def load_data(file_name):
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data

def get_all_cur_runs(run_dir, start, end, pkl_file):
	all_runs = []
	for i in range(start, end+1):
		run = load_data(run_dir + '/run_{}_all_data'.format(i) + '/{}'.format(pkl_file))
		all_runs.append(run)
	return np.array(all_runs)

def get_all_old_data(old_run_dir, start, end):
	all_old_runs = []
	for i in range(start, end+1):
		scores = load_data(old_run_dir + '/sc_pretrained_False_training_scores_{}.pkl'.format(i))
		all_old_runs.append(scores)
	return np.array(all_old_runs)

def get_multi_cur_data(run_dirs, start, end, is_old=False):
	multi_cur_data = {}
	for i, run_dir in enumerate(run_dirs):
		cur_scores = get_all_cur_runs(run_dir, start, end, 'per_episode_scores.pkl')
		args = get_all_cur_runs(run_dir, start, end, 'args.pkl')
		num_options_history = get_all_cur_runs(run_dir, start, end, 'num_options_history.pkl')
		temporal_full_chain_breaks = get_all_cur_runs(run_dir, start, end, 'temporal_full_chain_breaks.pkl')
		
		if is_old:
			label = 'all_cur_old_data_{}'.format(i)
		else:
			label = 'all_cur_data_nu_{}'.format(args[0].pes_nu)
		
		multi_cur_data[label] = {'run_dir' : run_dir,
							'cur_scores' : cur_scores,
								'args' : args,
								'num_options_history' : num_options_history,
								'temporal_full_chain_breaks' : temporal_full_chain_breaks}
	return multi_cur_data

def plt_old_multi_avg_learning_curves(plot_dir, all_data, mdp_env_name, file_name, ci=95):
	columns = ['time_steps', 'run_data']
	colors = sns.color_palette("deep")
	i=0
	num_runs = 0

	for data_name, data in all_data.items():
		for cur_name, cur_dict in data.items():
			# Export current data dict
			args = cur_dict['args'][0]
			run_dir = cur_dict['run_dir']
			cur_scores = cur_dict['cur_scores']

			# Set number of runs
			if num_runs == 0:
				num_runs, _ = cur_scores.shape
			
			# Plotting variables
			num_eps = len(cur_scores[0])
			time_steps = np.linspace(1, num_eps, num=num_eps)
			if 'fixed' in run_dir:
				label = 'Fixed learning'
			else:
				label = 'Continuous learning'

			# Append all runs
			df_data = pd.DataFrame(columns=columns)
			for run in cur_scores:
				df = pd.DataFrame(np.column_stack((time_steps, run)), columns=columns)
				df_data = df_data.append(df)

			sns.lineplot(x=columns[0], y=columns[1], data=df_data, label=label, ci=ci, color=colors[i])
			i+=1

		# else:	# Old tests w/ Akhil's format
		# 	# Plotting variables
		# 	num_eps = len(data[0])
		# 	time_steps = np.linspace(1, num_eps, num=num_eps)
		# 	label = 'Old DSC'

		# 	# Set number of runs
		# 	if num_runs == 0:
		# 		num_runs, _ = data.shape

		# 	# Append all runs
		# 	df_data = pd.DataFrame(columns=columns)
		# 	for run in data:
		# 		df = pd.DataFrame(np.column_stack((time_steps, run)), columns=columns)
		# 		df_data = df_data.append(df)

		# 	sns.lineplot(x=columns[0], y=columns[1], data=df_data, label=label, ci=ci, color=colors[-3])

	# Plotting features
	plt.title('Initiation Set Classifier (previous DSC): {} ({} runs)'.format(mdp_env_name, num_runs))
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

	# Save plots
	plt.savefig("{}/{}.png".format(plot_dir, file_name), bbox_inches='tight')
	plt.close()
	print("Plot {}/{}.png saved!".format(plot_dir, file_name))

def plot_multi_avg_learning_curves(plot_dir, all_data, mdp_env_name, file_name, ci=95, one_v_one=False, single_color_idx=None, with_breaks=False):
	columns = ['time_steps', 'run_data']
	colors = sns.color_palette("deep")
	i=0
	num_runs = 0

	for data_name, data in all_data.items():
		if 'cur' in data_name:	# Multiple current tests
			for cur_name, cur_dict in data.items():
				# Export current data dict
				args = cur_dict['args'][0]
				cur_scores = cur_dict['cur_scores']
				temporal_full_chain_breaks = cur_dict['temporal_full_chain_breaks']

				# Set number of runs
				if num_runs == 0:
					num_runs, _ = cur_scores.shape
				
				# Plotting variables
				num_eps = len(cur_scores[0])
				time_steps = np.linspace(1, num_eps, num=num_eps)
				if 'old' in data_name:
					label = 'Old DSC'
				else:
					label = 'DSC w/ Opt/Pes: nu={}'.format(args.pes_nu)

				# Append all runs
				df_data = pd.DataFrame(columns=columns)
				for run in cur_scores:
					df = pd.DataFrame(np.column_stack((time_steps, run)), columns=columns)
					df_data = df_data.append(df)

				if one_v_one and 'old' not in data_name:
					sns.lineplot(x=columns[0], y=columns[1], data=df_data, label=label, ci=ci, color=colors[single_color_idx])
				elif 'old' in data_name:
					sns.lineplot(x=columns[0], y=columns[1], data=df_data, label=label, ci=ci, color=colors[-3])
				else:
					sns.lineplot(x=columns[0], y=columns[1], data=df_data, label=label, ci=ci, color=colors[i])
					i+=1

				if with_breaks:
					print(temporal_full_chain_breaks)
					exit()

		else:	# Old tests w/ Akhil's format
			# Plotting variables
			num_eps = len(data[0])
			time_steps = np.linspace(1, num_eps, num=num_eps)
			label = 'Old DSC'

			# Set number of runs
			if num_runs == 0:
				num_runs, _ = data.shape

			# Append all runs
			df_data = pd.DataFrame(columns=columns)
			for run in data:
				df = pd.DataFrame(np.column_stack((time_steps, run)), columns=columns)
				df_data = df_data.append(df)

			sns.lineplot(x=columns[0], y=columns[1], data=df_data, label=label, ci=ci, color=colors[-3])

	# Plotting features
	plt.title('Average Reward: {} ({} runs)'.format(mdp_env_name, num_runs))
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

	# Save plots
	if one_v_one:
		plt.savefig("{}/{}.png".format(plot_dir, file_name), bbox_inches='tight')
		plt.close()
		print("Plot {}/{}.png saved!".format(plot_dir, file_name))
	else:
		plt.savefig("{}/{}.png".format(plot_dir, file_name), bbox_inches='tight')
		plt.close()
		print("Plot {}/{}.png saved!".format(plot_dir, file_name))

def plot_average_number_of_options(plot_dir, all_data, mdp_env_name, file_name, ci=None):
	columns = ['nu', 'num_options']
	sns.color_palette("deep")
	num_runs = 0

	df_data = pd.DataFrame(columns=columns)
	for data_name, data in all_data.items():
		for i, (cur_name, cur_dict) in enumerate(data.items()):
			# Export current data dict
			args = cur_dict['args']
			num_options_history = np.amax(cur_dict['num_options_history'], axis=1)

			# Set number of runs
			if num_runs == 0:
				num_runs = len(num_options_history)
			
			# Append all runs
			for run in num_options_history:
				df = pd.DataFrame(np.column_stack((args[i].pes_nu, run)), columns=columns)
				df_data = df_data.append(df)

	sns.barplot(x=columns[0], y=columns[1], data=df_data, ci=ci)

	# Plotting features
	plt.title('Average Number of Options: {} ({} runs)'.format(mdp_env_name, num_runs))
	plt.ylabel('Number of Options')
	plt.xlabel('Value of nu')

	# Save plots
	plt.savefig("{}/{}.png".format(plot_dir, file_name), bbox_inches='tight')
	plt.close()
	print("Plot {}/{}.png saved!".format(plot_dir, file_name))

def plot_average_reward_per_number_options(plot_dir, all_data, mdp_env_name, file_name, ci=None):
	columns = ['num_options', 'cum_reward']
	sns.color_palette("deep")
	num_runs = 0
	pes_nu = None

	df_data = pd.DataFrame(columns=columns)
	for data_name, data in all_data.items():
		for cur_name, cur_dict in data.items():
			# Export current data dict
			if pes_nu is None:
				pes_nu = cur_dict['args'][0].pes_nu
			num_options_history = np.amax(cur_dict['num_options_history'], axis=1)
			cur_scores = cur_dict['cur_scores']

			cum_rewards = np.sum(cur_scores, axis=1)

			# Set number of runs
			if num_runs == 0:
				num_runs = len(num_options_history)
			
			# Append all runs
			for num_option, cum_reward in zip(num_options_history, cum_rewards):
				df = pd.DataFrame(np.column_stack((num_option, cum_reward)), columns=columns)
				df_data = df_data.append(df)
	
	sns.barplot(x=columns[0], y=columns[1], data=df_data, ci=ci)
	
	# Plotting features
	plt.title('Average cumulative reward per option: {} (nu = {}, {} runs)'.format(mdp_env_name, pes_nu, num_runs))
	plt.xlabel('Number of Options')
	plt.ylabel('Average Cumulative Reward')
	
	# Save plots
	plt.savefig("{}/{}.png".format(plot_dir, file_name), bbox_inches='tight')
	plt.close()
	print("Plot {}/{}.png saved!".format(plot_dir, file_name))

def main(args):
	# Create plotting dir (if not created)
	path = 'plots'
	Path(path).mkdir(exist_ok=True)

	# (Akhil) Old comparative data
	# all_old_data = get_all_old_data(args.old_run_dir, 1, 10)

	# Multi old data
	all_old_data = get_multi_cur_data(args.old_cur_run_dirs, 1, 10, is_old=True)

	# Multi hyperparameter data
	multi_cur_data = get_multi_cur_data(args.run_dirs, 1, 10)

	# nus = [0.1]
	# all_old_data = {'all_cur_old_data' : all_old_data['all_cur_old_data']} 	# TODO: use for cur old format
	# for idx, nu in enumerate(nus):
	# 	name = 'all_cur_data_nu_{}'.format(nu)
	# 	single_cur_data = {name : multi_cur_data[name]}

	# 	# all_data = {'single_cur_data' : single_cur_data, 'all_cur_old_data' : all_old_data}
	# 	# file_name = 'maze_chain_fix_nu_{}_v_old_learning_curves'.format(nu)
		
	# 	# all_data = {'single_cur_data' : single_cur_data}
	# 	# file_name = 'maze_chain_fix_pes_nu_{}_learning_curves'.format(nu)

	# 	all_data = {'all_cur_old_data' : all_old_data}
	# 	file_name = 'maze_old_fixed_v_continous_learning_learning_curves'

	# 	plot_multi_avg_learning_curves(plot_dir=path,
	# 								all_data=all_data,
	# 								mdp_env_name=args.mdp_env_name,
	# 								# file_name='maze_chain_fix_nu_{}_v_old_learning_curves'.format(nu),
	# 								file_name=file_name,
	# 								ci='sd',
	# 								one_v_one=True,
	# 								single_color_idx=idx)

	# Plot full chain breaks overlapped reward
	# all_data = {'all_cur_old_data' : all_old_data}
	# plot_multi_avg_learning_curves(plot_dir=path,
	# 							   all_data=all_data,
	# 							   mdp_env_name=args.mdp_env_name,
	# 							   file_name='old_learning_curve_with_breaks',
	# 							   ci='sd',
	# 							   with_breaks=True)

	# For all hyperparameters, plot average reward for number of options
	# nus = [0.5, 0.6, 0.7, 0.8]
	# for idx, nu in enumerate(nus):
	# 	name = 'all_cur_data_nu_{}'.format(nu)
	# 	single_cur_data = {name : multi_cur_data[name]}
	# 	all_data = {'single_cur_data' : single_cur_data}

	# 	plot_average_reward_per_number_options(plot_dir=path,
	# 										   all_data=all_data,
	# 										   mdp_env_name=args.mdp_env_name,
	# 										   file_name="maze_chain_fix_nu_{}_reward_per_option".format(nu),
	# 										   ci='sd')

	# Plot average number of options per hyperparameter
	# all_data = {'multi_cur_data' : multi_cur_data}
	# plot_average_number_of_options(plot_dir=path,
	# 								all_data=all_data,
	# 								mdp_env_name=args.mdp_env_name,
	# 								file_name='maze_old_fixed_v_continous_learning_learning_curves',
	# 								ci='sd')

	# Plot old runs
	# all_data = {'all_old_data' : all_old_data}
	# plot_multi_avg_learning_curves(plot_dir=path,
	# 							   all_data=all_data,
	# 							   mdp_env_name=args.mdp_env_name,
	# 							   file_name='old_learning_curve')
	
	# Plot single hyperparameter
	# all_data = {'multi_cur_data' : multi_cur_data}
	# plot_multi_avg_learning_curves(plot_dir=path,
	# 							   all_data=all_data,
	# 							   mdp_env_name=args.mdp_env_name,
	# 							   file_name='single_learning_curve_chain_break_nu=0.6',
	# 							   ci='sd')

	# Plot old multiple runs
	all_data = {'multi_cur_old_data' : all_old_data}
	plt_old_multi_avg_learning_curves(plot_dir=path,
										all_data=all_data,
										mdp_env_name=args.mdp_env_name,
										file_name='maze_old_fixed_v_continous_learning_curves',
										ci='sd')	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_dirs', nargs='*', type=str, default="")
	parser.add_argument('--old_run_dir', type=str, default="")
	parser.add_argument('--mdp_env_name', type=str, default="")
	parser.add_argument('--old_cur_run_dirs', nargs='*', type=str, default="")
	args = parser.parse_args()

	main(args)