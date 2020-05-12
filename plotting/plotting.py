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
import matplotlib.colors as mc

sns.set(color_codes=True)
sns.set_style("white")

def rgba(r, g, b, a):
	return (r/255, g/255, b/255, a)

def adjust_lightness(color, amount=0.5):
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def make_meshgrid(x, y, h=.01):
		"""Create a mesh of points to plot in

		Args:
			x: data to base x-axis meshgrid on
			y: data to base y-axis meshgrid on
			h: stepsize for meshgrid, optional

		Returns:
			X and y mesh grid
		"""
		x_min, x_max = x.min(), x.max()
		y_min, y_max = y.min(), y.max()
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
		return xx, yy

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
		if 'pessimistic' in clf_name.lower():
			cf = plt.contourf(x_mesh, y_mesh, z, colors=color, alpha=alpha, hatches='xxx')
			patches.append(mpatches.Patch(color=color, label=clf_name, alpha=alpha, hatch='xxx'))
		else:
			cf = plt.contourf(x_mesh, y_mesh, z, colors=color, alpha=alpha)
			patches.append(mpatches.Patch(color=color, label=clf_name, alpha=alpha))

	plt.xticks(())
	plt.yticks(())
	plt.title("{}".format(option_name))
	plt.legend(handles=patches, 
				bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	
	# Save plots
	plt.savefig("{}/{}_{}.png".format(path, option_name, episode),bbox_inches='tight', edgecolor='black', format='png', quality=100)
	plt.close()

	# TODO: remove
	print("Figure {}/{}_{}.png saved!".format(path, option_name, episode))

def plot_multi_boundaries(x_mesh, y_mesh, option_data, rgb_color_palette, experiment_name, alpha, plot_eps, title, img_name=None, img_alpha=1, goal=None, start=None, pes_plots=True):
	# Create plotting dir (if not created)
	path = '{}/plots/clf_plots'.format(experiment_name)
	Path(path).mkdir(exist_ok=True)

	if img_name:
		back_img = plt.imread(img_name)
		plt.imshow(back_img, extent=[x_mesh.min(), x_mesh.max(), y_mesh.min(), y_mesh.max()], alpha=img_alpha)

	patches = []
	saved_eps = None

	# Gray classifier key
	default_colors = ['lightgrey', 'grey']
	once = False
	for i, (option_name, option) in enumerate(option_data.items()):
		if not once:
			for episode, episode_data in option.items():
				once = True
				clfs = episode_data['clfs_bounds']
				if episode == plot_eps:
					saved_eps = episode
					for (clf_name, clf), color in zip(clfs.items(), default_colors):
						if 'pessimistic' in clf_name.lower():
							if pes_plots:
								patches.append(mpatches.Patch(color=color, label='{}'.format(clf_name), alpha=alpha, hatch='xxx'))
						elif 'Initiation' in clf_name:
							patches.append(mpatches.Patch(color=color, label='Initiation set', alpha=alpha))
						else:
							patches.append(mpatches.Patch(color=color, label='{}'.format(clf_name), alpha=alpha))

	colors =[mc.to_hex(rgb) for rgb in rgb_color_palette]
	dark_colors = [mc.to_hex(adjust_lightness(color, amount=0.5)) for color in colors]

	for i, (option_name, option) in enumerate(option_data.items()):
		for episode, episode_data in option.items():
			clfs = episode_data['clfs_bounds']
			if episode == plot_eps:
				saved_eps = episode
				patches.append(mpatches.Patch(color=colors[i], label='{}'.format(option_name), alpha=alpha))

				# Plot classifier boundaries
				for (clf_name, clf) in clfs.items():
					if 'pessimistic' in clf_name.lower():
						if pes_plots:
							z = clf.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
							z = (z.reshape(x_mesh.shape) > 0).astype(int)
							z = np.ma.masked_where(z == 0, z)
							cf = plt.contourf(x_mesh, y_mesh, z, colors=dark_colors[i], alpha=alpha, hatches='xxx')
							for collection in cf.collections:
								collection.set_edgecolor(dark_colors[i])
					else:
						z = clf.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
						z = (z.reshape(x_mesh.shape) > 0).astype(int)
						z = np.ma.masked_where(z == 0, z)
						cf = plt.contourf(x_mesh, y_mesh, z, colors=colors[i] , alpha=alpha-0.2)

	g = plt.scatter(goal[0], goal[1], marker="*", color="y", s=80, path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label="Goal")
	patches.append(g)
		
	s = plt.scatter(start[0], start[1], marker="X", color="k", label="Start")
	patches.append(s)

	plt.xticks(())
	plt.yticks(())
	plt.title(title)
	plt.legend(handles=patches, 
				bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	
	# Save plots
	plt.savefig("{}/all_options_{}.png".format(path, saved_eps),bbox_inches='tight', edgecolor='black', format='png', quality=100)
	plt.close()

	print("Figure {}/all_options_{}.png saved!".format(path, saved_eps))

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

def generate_all_plots(run_dir, option_data, per_episode_scores, x_mesh, y_mesh, mdp_env_name, num_run, args, all_cur_data=None, all_old_data=None, multi_cur_data=None, img_dir=None, goal=None, start=None):
	sns.set_style("white")
	
	# Plot actual boundaries
	# for i, (option_name, option) in enumerate(option_data.items()):
	# 	# dont_plot = ['overall_goal_policy_option', 'option_1']
	# 	dont_plot = []
	# 	# print(option)
	# 	if option_name not in dont_plot:
	# 		for episode, episode_data in option.items():
	# 			colors = [all_color_strs[i], 'dark' + all_color_strs[i]]
	# 			clfs_bounds = episode_data['clfs_bounds']
	# 			# plot boundaries of classifiers
	# 			if (episode % 299 == 0):
	# 				plot_boundary(x_mesh=x_mesh,
	# 							y_mesh=y_mesh,
	# 							clfs=clfs_bounds,
	# 							colors=colors,
	# 							option_name=option_name,
	# 							episode=episode,
	# 							experiment_name=run_dir,
	# 							alpha=0.5,
	# 							img_name=img_dir,
	# 							goal=goal,	
	# 							start=start)

	# Plot overlap boundaries
	start_eps, end_eps, num_samples = 42, 43, 1
	episodes = np.linspace(start_eps, end_eps, num_samples, dtype=int)
	rgb_color_palette = sns.color_palette('deep')

	for plot_eps in episodes:
		plot_multi_boundaries(x_mesh=x_mesh,
									y_mesh=y_mesh,
									option_data=option_data,
									rgb_color_palette=rgb_color_palette,
									experiment_name=run_dir,
									alpha=0.8,
									plot_eps=plot_eps,
									title='Chain Fix Methods',
									img_name=img_dir,
									img_alpha=0.95,
									goal=goal,
									start=start,
									pes_plots=True)

def export_to_text_file(data, file_dir, file_name):
	with open(file_dir + '/' + file_name, 'w') as f:
		f.write(str(data))
	print("File {} saved!".format(file_dir + '/' + file_name))

def main(args):
	run_dir = args.run_dir
	data_dir = args.data_dir
	img_dir = args.img_dir
	
	# Load variables
	option_data = load_data(run_dir + '/' + data_dir + '/option_data.pkl')
	per_episode_scores = load_data(run_dir + '/' + data_dir + '/per_episode_scores.pkl')
	mdp_env_name = load_data(run_dir + '/' + data_dir + '/mdp_env_name.pkl')
	args = load_data(run_dir + '/' + data_dir + '/args.pkl')
	skill_chain = load_data(run_dir + '/' + data_dir + '/final_skill_chain.pkl')
	
	x_mesh = load_data(run_dir + '/' + data_dir + '/x_mesh.pkl')
	y_mesh = load_data(run_dir + '/' + data_dir + '/y_mesh.pkl')
	
	# Point maze
	# width_coord = height_coord = np.array([-2,11])
	# x_mesh, y_mesh = make_meshgrid(width_coord, height_coord, h=0.01)
	
	goal_xy = load_data(run_dir + '/' + data_dir + '/goal_xy.pkl')
	start_xy = load_data(run_dir + '/' + data_dir + '/start_xy.pkl')

	# Create plotting directory
	Path(run_dir + '/' + data_dir + '/plots').mkdir(exist_ok=True)

	# Export skill chain
	export_to_text_file(skill_chain, run_dir + '/' + data_dir, 'skill_chain.txt')

	# Generate plots
	generate_all_plots(run_dir=run_dir + '/' + data_dir,
					   option_data=option_data,
					   per_episode_scores=per_episode_scores,
					   x_mesh=x_mesh,
					   y_mesh=y_mesh,
					   mdp_env_name=mdp_env_name,
					   num_run=args.num_run,
					   args=args,
					   img_dir=img_dir,
					   goal=goal_xy,
					   start=start_xy)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_dir', type=str, default="")
	parser.add_argument('--data_dir', type=str, default="")
	parser.add_argument('--img_dir', type=str, default="")
	args = parser.parse_args()

	main(args)