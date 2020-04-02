import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

def load_data(file_name):
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data

def plot_boundary(x_mesh, y_mesh, X_pos, clfs, colors, option_name, episode, experiment_name, alpha):
	# Create plotting dir (if not created)
	path = '{}/plots/clf_plots'.format(experiment_name)
	Path(path).mkdir(exist_ok=True)

	patches = []

	# Plot classifier boundaries
	for (clf_name, clf), color in zip(clfs.items(), colors):
		z = clf.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
		z = (z.reshape(x_mesh.shape) > 0).astype(int)
		z = np.ma.masked_where(z == 0, z)

		cf = plt.contourf(x_mesh, y_mesh, z, colors=color, alpha=alpha)
		patches.append(mpatches.Patch(color=color, label=clf_name, alpha=alpha))

	cb = plt.colorbar()
	cb.remove()

	# Plot successful trajectories
	x_pos_coord, y_pos_coord = X_pos[:,0], X_pos[:,1]
	pos_color = 'black'
	p = plt.scatter(x_pos_coord, y_pos_coord, marker='+', c=pos_color, label='successful trajectories', alpha=alpha)
	patches.append(p)

	plt.xticks(())
	plt.yticks(())
	plt.title("Classifier Boundaries - {}".format(option_name))
	plt.legend(handles=patches, 
				bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	
	# Save plots
	plt.savefig("{}/{}_{}.png".format(path, option_name, episode),bbox_inches='tight', edgecolor='black')
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

def plot_learning_curves(experiment_name, data, mdp_env_name):
	# Create plotting dir (if not created)
	path = '{}/plots/learning_curves'.format(experiment_name)
	Path(path).mkdir(exist_ok=True)
	
	plt.plot(data, label=experiment_name)
		
	plt.title("{}".format(mdp_env_name))
	plt.ylabel("Rewards")
	plt.xlabel("Episodes")
	plt.legend(title="Tests", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

	# Save plots
	plt.savefig("{}/learning_curve.png".format(path), bbox_inches='tight')
	plt.close()

	# TODO: remove
	print("|-> {}/learning_curve.png saved!".format(path))

def generate_all_plots(experiment_name, all_clf_probs, option_data, per_episode_scores, x_mesh, y_mesh, mdp_env_name):
	sns.set_style("white")
	num_colors = 100
	colors = ['blue', 'green']
	cmaps = [cm.get_cmap('Blues', num_colors), cm.get_cmap('Greens', num_colors)]

	for option_name, option in option_data.items():
		print("Generating plots for {}..".format(option_name))
		for episode, episode_data in option.items():
			clfs_bounds = episode_data['clfs_bounds']
			clfs_probs = episode_data['clfs_probs']
			X_pos = episode_data['X_pos']

			# plot boundaries of classifiers
			plot_boundary(x_mesh=x_mesh,
						  y_mesh=y_mesh,
						  X_pos=X_pos,
						  clfs=clfs_bounds,
						  colors=colors,
						  option_name=option_name,
						  episode=episode,
						  experiment_name=experiment_name,
						  alpha=0.5)

			# plot state probability estimates
			# plot_state_probs(x_mesh=x_mesh,
			# 				 y_mesh=y_mesh,
			# 				 clfs=clfs_probs,
			# 				 option_name=option_name,
			# 				 cmaps=cmaps,
			# 				 episode=episode,
			# 				 experiment_name=experiment_name)
		print()
		
	# plot average probabilities
	# plot_prob(all_clf_probs=all_clf_probs,
	# 		  experiment_name=experiment_name)

	# plot learning curves
	plot_learning_curves(experiment_name=experiment_name,
						 data=per_episode_scores,
						 mdp_env_name=mdp_env_name)

def main(run_dir, data_dir):

	# Load variables
	experiment_name = load_data(run_dir + '/' + data_dir + '/experiment_name.pkl')
	all_clf_probs = load_data(run_dir + '/' + data_dir + '/all_clf_probs.pkl')
	option_data = load_data(run_dir + '/' + data_dir + '/option_data.pkl')
	per_episode_scores = load_data(run_dir + '/' + data_dir + '/per_episode_scores.pkl')
	x_mesh = load_data(run_dir + '/' + data_dir + '/x_mesh.pkl')
	y_mesh = load_data(run_dir + '/' + data_dir + '/y_mesh.pkl')
	mdp_env_name = load_data(run_dir + '/' + data_dir + '/mdp_env_name.pkl')

	# Create plotting directory
	Path(run_dir + '/plots').mkdir(exist_ok=True)

	# Generate plots
	generate_all_plots(experiment_name=experiment_name,
					   all_clf_probs=all_clf_probs,
					   option_data=option_data,
					   per_episode_scores=per_episode_scores,
					   x_mesh=x_mesh,
					   y_mesh=y_mesh,
					   mdp_env_name=mdp_env_name)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_dir', type=str, default="")
	parser.add_argument('--data_dir', type=str, default="")
	args = parser.parse_args()

	main(run_dir=args.run_dir, data_dir=args.data_dir)