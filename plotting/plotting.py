import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.patheffects as pe

sns.set(color_codes=True)
sns.set_style("white")

def load_data(file_name):
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data

def plot_boundary(x_mesh, y_mesh, clfs, colors, option_name, episode, experiment_name, alpha, img_name=None, goal=None, start=None):
	# Create plotting dir (if not created)
	path = '{}/plots/clf_plots'.format(experiment_name)
	Path(path).mkdir(exist_ok=True)

	if img_name:
		back_img = plt.imread(img_name)
		plt.imshow(back_img, extent=[x_mesh.min(), x_mesh.max(), y_mesh.min(), y_mesh.max()], alpha=0.3)

	patches = []

	g = plt.scatter(goal[0], goal[1], marker="*", color="y", s=80, path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label="Goal")
	patches.append(g)
		
	s = plt.scatter(start[0], start[1], marker="X", color="k", label="Start")
	patches.append(s)


	# Plot classifier boundaries
	for (clf_name, clf), color in zip(clfs.items(), colors):
		z = clf.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
		z = (z.reshape(x_mesh.shape) > 0).astype(int)
		z = np.ma.masked_where(z == 0, z)

		cf = plt.contourf(x_mesh, y_mesh, z, colors=color, alpha=alpha)
		patches.append(mpatches.Patch(color=color, label=clf_name, alpha=alpha))

	cb = plt.colorbar()
	cb.remove()

	plt.xticks(())
	plt.yticks(())
	plt.title("{}".format(option_name))
	plt.legend(handles=patches, 
				bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	
	# Save plots
	plt.savefig("{}/{}_{}.png".format(path, option_name, episode),bbox_inches='tight', edgecolor='black', format='png', quality=100)
	plt.close()

	# TODO: remove
	print("|-> {}/{}_{}.png saved!".format(path, option_name, episode))

def plot_state_probs(x_mesh, y_mesh, clfs, option_name, cmaps, episode, experiment_name):
	# Create plotting dir (if not created)
	path = '{}/plots/state_prob_plots'.format(experiment_name)
	Path(path).mkdir(exist_ok=True)
	
	# Plot state probability estimates
	for (clf_name, clf), cmap in zip(clfs.items(), cmaps):

		fig, ax = plt.subplots()
		
		# Only plot positive predictions
		X_mesh = np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T
		probs = clf.predict_proba(X_mesh)[:, 1]
		preds = (clf.predict(X_mesh) > 0).astype(int)
		probs = np.multiply(probs, preds)

		ax.set_title(
			'Probability Estimates ({}) - {}'.format(clf_name, option_name))
		
		states = ax.pcolormesh(x_mesh, y_mesh, probs.reshape(x_mesh.shape), shading='gouraud', cmap=cmap, vmin=0.0, vmax=1.0)
		cbar = fig.colorbar(states)

		ax.set_xticks(())
		ax.set_yticks(())
		
		# Save plots
		plt.savefig("{}/{}_{}_{}.png".format(path, option_name,
					clf_name, episode), bbox_inches='tight', edgecolor='black')
		plt.close()

		# TODO: remove
		print("|-> {}/{}_{}_{}.png saved!".format(path, option_name, clf_name, episode))

def plot_prob(all_clf_probs, experiment_name):
	# Create plotting dir (if not created)
	path = '{}/plots/prob_plots'.format(experiment_name)
	Path(path).mkdir(exist_ok=True)

	for clf_name, clf_probs in all_clf_probs.items():
		colors = sns.hls_palette(len(clf_probs), l=0.5)
		for i, (opt_name, opt_probs) in enumerate(clf_probs.items()):
			probs, episodes = opt_probs
			if 'optimistic' in clf_name:
				plt.plot(episodes, probs, c=colors[i], label="{} - {}".format(clf_name, opt_name))
			else:
				plt.plot(episodes, probs, c=colors[i], label="{} - {}".format(clf_name, opt_name), linestyle='--')

	plt.title("Average Probability Estimates")
	plt.ylabel("Probabilities")
	plt.xlabel("Episodes")
	plt.legend(title="Options", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

	# Save plots
	plt.savefig("{}/option_prob_estimates.png".format(path), bbox_inches='tight')
	plt.close()
	
	# TODO: remove
	print("|-> {}/option_prob_estmates.png saved!".format(path))

def plot_learning_curve(experiment_name, data, mdp_env_name, num_run):
	# Create plotting dir (if not created)
	path = '{}/plots/learning_curves'.format(experiment_name)
	Path(path).mkdir(exist_ok=True)
	
	plt.plot(data, label=experiment_name)
		
	plt.title("{}".format(mdp_env_name))
	plt.ylabel("Rewards")
	plt.xlabel("Episodes")
	plt.legend(title="Tests", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

	# Save plots
	plt.savefig("{}/learning_curve_{}.png".format(path, num_run), bbox_inches='tight')
	plt.close()

	# TODO: remove
	print("Plot {}/{}_learning_curve.png saved!".format(path, num_run))

def get_all_cur_data(run_dir, start, end):
	all_data = []
	for i in range(start, end+1):
		scores = load_data(run_dir + '/run_{}_all_data'.format(i) + '/per_episode_scores.pkl')
		all_data.append(scores)
	return np.array(all_data)

def get_all_old_data(old_run_dir, start, end):
	all_old_runs = []
	for i in range(start, end+1):
		scores = load_data(old_run_dir + '/sc_pretrained_False_training_scores_{}.pkl'.format(i))
		all_old_runs.append(scores)
	return np.array(all_old_runs)

def get_avg_data(all_data):
	return np.sum(all_data, axis=0) / len(all_data)

def plot_avg_learning_curves(experiment_name, all_data, mdp_env_name, args):
	# Create plotting dir (if not created)
	path = '{}/plots/learning_curves'.format(experiment_name)
	Path(path).mkdir(exist_ok=True)

	columns = ['time_steps', 'run_data']

	for data_name, data in all_data.items():
		num_eps = len(data[0])
		time_steps = np.linspace(1, num_eps, num=num_eps)

		# Method labels
		if 'all_cur_data' in data_name:
			label = 'DSC w/ Opt/Pes: nu={}'.format(args.nu)
		else:
			label = 'DSC (old)'

		# Append all runs
		df_all_data = pd.DataFrame(columns=columns)
		for run in data:
			df = pd.DataFrame(np.column_stack((time_steps, run)), columns=columns)
			df_all_data = df_all_data.append(df)

		sns.lineplot(x='time_steps', y='run_data', data=df_all_data, label=label)		

	plt.title('Average Reward ({} runs): {}'.format(len(data), mdp_env_name))
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

	# num_eps = len(old_data[0])
	# time_steps = np.linspace(1,num_eps,num=num_eps)

	# all_old_data = pd.DataFrame(columns=columns)
	# for run in old_data:
	# 	df = pd.DataFrame(np.column_stack((time_steps, run)), columns=columns)
	# 	all_old_data = all_data.append(df)


	# ax = sns.lineplot(x='time_steps', y='run_data', data=all_data, label=label)
	# sns.lineplot(x='time_steps', y='run_data', data=all_old_data, label='DSC (old)')

	# ax.set(xlabel='Episodes', ylabel='Rewards', title='Average Reward ({} runs): {}'.format(len(data), mdp_env_name))
	# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

	# Save plots
	plt.savefig("{}/average_learning_curves.png".format(path), bbox_inches='tight')
	plt.close()

	# TODO: remove
	print("Plot {}/average_learning_curves.png saved!".format(path))

def generate_all_plots(run_dir, option_data, per_episode_scores, x_mesh, y_mesh, mdp_env_name, num_run, args, all_cur_data=None, all_old_data=None, multi_cur_data=None, img_dir=None):
	sns.set_style("white")
	# num_colors = 100
	all_color_strs =['salmon', 'green', 'orange', 'cyan', 'khaki', 'slateblue', 'goldenrod', 'turquoise', 'red', 'blue', 'slategray', 'violet', 'magenta']
	# cmaps = [cm.get_cmap('Blues', num_colors), cm.get_cmap('Greens', num_colors)]

	# print(option_data)

	for i, (option_name, option) in enumerate(option_data.items()):
			
		# dont_plot = ['overall_goal_policy_option', 'option_1']
		dont_plot = []
		# print(option)
		if option_name not in dont_plot:
			for episode, episode_data in option.items():
				colors = [all_color_strs[i], 'dark' + all_color_strs[i]]
				clfs_bounds = episode_data['clfs_bounds']
				# plot boundaries of classifiers
				if (episode % 299 == 0):
					plot_boundary(x_mesh=x_mesh,
								y_mesh=y_mesh,
								clfs=clfs_bounds,
								colors=colors,
								option_name=option_name,
								episode=episode,
								experiment_name=run_dir,
								alpha=0.5,
								img_name=img_dir,
								goal=np.array([0.0, 8.0]),	# Fix hardcode
								start=np.array([-0.08985623, -0.097918]))

	# plot single learning curve
	# plot_learning_curve(experiment_name=run_dir,
	# 					data=per_episode_scores,
	# 					mdp_env_name=mdp_env_name,
	# 					num_run=num_run)

	# plot cur vs old average learning curves
	# all_data = {'all_cur_data': all_cur_data, 'all_old_data' : all_old_data}
	# all_data = {'all_old_data' : all_old_data}
	# plot_avg_learning_curves(experiment_name=run_dir,
	# 				all_data=all_data,
	# 				mdp_env_name=mdp_env_name,
	# 				args=args)

def main(args):
	run_dir = args.run_dir
	data_dir = args.data_dir
	img_dir = args.img_dir
	# old_run_dir = args.old_run_dir
	
	# Load variables
	# experiment_name = load_data(run_dir + '/' + data_dir + '/experiment_name.pkl')
	option_data = load_data(run_dir + '/' + data_dir + '/option_data.pkl')
	per_episode_scores = load_data(run_dir + '/' + data_dir + '/per_episode_scores.pkl')
	x_mesh = load_data(run_dir + '/' + data_dir + '/x_mesh.pkl')
	y_mesh = load_data(run_dir + '/' + data_dir + '/y_mesh.pkl')
	mdp_env_name = load_data(run_dir + '/' + data_dir + '/mdp_env_name.pkl')
	args = load_data(run_dir + '/' + data_dir + '/args.pkl')
	# Multiple score data
	# all_old_data = get_all_old_data(old_run_dir, 1, 20)
	# all_cur_data = get_all_cur_data(run_dir, 1, 10)

	# Create plotting directory
	Path(run_dir + '/plots').mkdir(exist_ok=True)

	# Generate plots
	generate_all_plots(run_dir=run_dir,
					   option_data=option_data,
					   per_episode_scores=per_episode_scores,
					   x_mesh=x_mesh,
					   y_mesh=y_mesh,
					   mdp_env_name=mdp_env_name,
					   num_run=args.num_run,
					   args=args,
					   img_dir=img_dir)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_dir', type=str, default="")
	# parser.add_argument('--old_run_dir', type=str, default="None")
	parser.add_argument('--data_dir', type=str, default="")
	parser.add_argument('--img_dir', type=str, default="")
	args = parser.parse_args()

	main(args)