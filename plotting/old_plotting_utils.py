import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import pdb
import seaborn as sns
sns.set()


def get_plot_params(array):
    median = np.median(array, axis=0)
    means = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    top = means + std
    bot = means - std
    return median, means, top, bot


def moving_average(a, n=25):
 ret = np.cumsum(a, dtype=float)
 ret[n:] = ret[n:] - ret[:-n]
 return ret[n-1:] / n


def smoothen_data(scores, n=10):
	print(scores.shape)
	smoothened_cols = scores.shape[1] - n + 1
	smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
	for i in range(scores.shape[0]):
		smoothened_data[i, :] = moving_average(scores[i, :], n=n)
	return smoothened_data


def generate_plot(score_array, label):
	smoothened_data = smoothen_data(score_array)
	median, mean, top, bottom = get_plot_params(smoothened_data)
	plt.plot(median, linewidth=2, label=label, alpha=0.9)
	plt.fill_between( range(len(top)), top, bottom, alpha=0.2 )


def get_top_k(scores, k=6):
    cumulative_scores = np.sum(scores, axis=1)
    sorted_indices = np.argsort(cumulative_scores)
    top_k_indices = sorted_indices[-k:]
    return scores[top_k_indices, :]


def get_scores(log_dir, mode):
	overall_scores = []
	if "ddpg" in log_dir:
		glob_pattern = log_dir + "/" + "*_{}_scores_*".format(mode)
	else:
		glob_pattern = log_dir + "/" + "*_{}_scores_*".format(mode)
	for score_file in glob.glob(glob_pattern):
	    print(score_file)
	    with open(score_file, "rb") as f:
	        scores = pickle.load(f)
	    overall_scores.append(scores)
	score_array = np.array(overall_scores)
	return score_array


def create_plots(sc_log_dir, hac_log_dir, domain_name):
	plt.figure(figsize=(8, 6))
	for i, mode in enumerate(["training", "validation"]):
		plt.subplot(1, 2, i+1)
		if sc_log_dir: 
			sc_array = get_scores(sc_log_dir, mode)
			generate_plot(sc_array, label="DSC (Ours)")
		
		if hac_log_dir and mode == "validation":
			hac_array = get_scores(hac_log_dir, mode)
			generate_plot(hac_array, label="HAC")

		if args.ddpg_log_dir and mode == "training":
			ddpg_array = get_scores(args.ddpg_log_dir, mode)
			generate_plot(ddpg_array, label="DDPG (Flat)")

		if args.ppoc_log_dir and mode == "validation":
			ppoc_array = get_scores(args.ppoc_log_dir, mode)
			generate_plot(ppoc_array, label="Option-Critic")

		plt.xlabel("Episode")
		if i == 0:
			plt.ylabel("Reward")
			plt.legend(loc="upper left")
		else:
			plt.legend(loc='upper left')
		# plt.xlim((0, 2000))
	plt.suptitle(domain_name)
	plt.savefig(domain_name + "_comparison_curves.svg", dpi=1000)
	plt.close()

def create_option_comparison_plots(sc_log_dirs, ddpg_log_dirs, ppoc_log_dirs, domain_names):
	mode = "training"
	plt.figure(figsize=(8, 6))
	for i, (sc_log_dir, ddpg_log_dir, ppoc_log_dir, domain_name) in enumerate(zip(sc_log_dirs, ddpg_log_dirs, ppoc_log_dirs, domain_names)):
		plt.subplot(1, 2, i + 1)
		sc_array = get_scores(sc_log_dir, mode)
		generate_plot(sc_array, label="DSC (Ours)")
		ddpg_array = get_scores(ddpg_log_dir, mode)
		if domain_name == "Four Room":
			ddpg_array = ddpg_array[:, :750]
		generate_plot(ddpg_array, label="DDPG (Flat)")
		ppoc_array = get_scores(ppoc_log_dir, mode)
		generate_plot(ppoc_array, label="Option-Critic")
		plt.xlabel("Episode")
		if i == 0:
			#plt.ylim((-2000, 0))
			plt.ylabel("Reward")
			plt.legend(loc="bottom left")
		#else:
		#	plt.legend(loc='upper left')
		plt.title(domain_name)
	plt.savefig("option_comparison_curves.svg", dp1=1000)
	plt.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--sc_log_dir", type=str, default="")
	parser.add_argument("--hac_log_dir", type=str, default="")
	parser.add_argument("--ppoc_log_dir", type=str, default="")
	parser.add_argument("--ddpg_log_dir", type=str, default="")
	parser.add_argument("--env_name", type=str, default="")
	args = parser.parse_args()

	# create_plots(args.sc_log_dir, None, args.env_name)
	
	mode = ["training"]
	for i, mode in enumerate(mode):
		sc_array = get_scores(args.sc_log_dir, mode)
		# smoothen_data = smoothen_data(sc_array)
		median, mean, top, bottom =  get_plot_params(sc_array)
		
		plt.plot(median, linewidth=2, label="DSC (Ours)", alpha=0.9)
		plt.fill_between(range(len(top)), top, bottom, alpha=0.2)
		plt.xlabel("Episode")
		plt.ylabel("Reward")
		plt.legend(loc="upper left")
		plt.suptitle(args.env_name)
		plt.show()

	# TODO: test
	# file_path = args.sc_log_dir + '/' + '*_training_scores_*'
	# with open()
