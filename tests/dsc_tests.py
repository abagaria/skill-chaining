import pickle
import numpy as np
import datetime

def load_data(file_name):
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data

class ClassTest(object):

	def __init__(self, test_num, test_list):
		self.test_num = test_num
		self.test_list = test_list	

	def update_test_num(self, test_num):
		self.test_num = test_num

def get_skill_chain():

	# Create option dict
	# parent_child_dict = {option.parent.name : option.name for option in self.trained_options}
	parent_child_dict = {None: "option_3", "option_4" : "option_2", "option_1" : "option_4", "option_3" : "option_1"}
	num_options = len(parent_child_dict)

	skill_chain = [parent_child_dict[None]]
	while len(skill_chain) != num_options:
		last_name = skill_chain[-1]
		skill_chain.append(parent_child_dict[last_name])

	print(skill_chain)

def hms_to_s(hms_array):
	h, m, s = hms_array
	return s + (m * 60) + (h * 3600)

def runtime_stats(runtimes):
	# convert to seconds
	total_runtimes = []
	for runtime in runtimes:
		total_runtimes.append(hms_to_s(runtime))
	total_runtimes = np.array(total_runtimes)

	# return stats in hh:mm:ss format
	mean = datetime.timedelta(seconds=np.mean(total_runtimes))
	std = datetime.timedelta(seconds=np.std(total_runtimes))
	return mean, std

def compute_runtimes():
	olds = np.array([[2, 36,49],
					[0, 56, 50],
					[1, 10, 24],
					[0, 48, 2],
					[2, 13, 46],
					[1, 58, 4],
					[1, 25, 16],
					[1, 32, 36],
					[1, 5, 50],
					[1, 38, 32]])

	robusts = np.array([[1, 52, 4],
					   [1, 18, 52],
					   [1, 0, 10],
					   [2, 56, 37],
					   [1, 22, 16],
					   [1, 42, 44],
					   [1, 21, 3],
					   [2, 3, 16],
					   [1, 35, 38],
					   [0, 44, 53]])

	old_mean, old_std = runtime_stats(olds)
	robust_mean, robust_std = runtime_stats(robusts)

	print("OLD: {} +- {}".format(old_mean, old_std))
	print("ROBUST: {} +- {}".format(robust_mean, robust_std))

def main():
	# test_data = load_data('runs/(cs2951x) maze_old_continuous_learning_2/run_1_all_data/num_options_history.pkl')
	# last_option, idx = 0, None
	# for i,  option in enumerate(test_data):
	# 	if option > last_option:
	# 		last_option, idx = option, i
	
	# print(last_option, idx)
	
	test_data = load_data('/home/mjs/Projects/skill-chaining/saved_runs/final_option_1_ddpg_agent_replay_buffer.pkl')
	print(test_data)


	# compute_runtimes()
	


if __name__ == "__main__":
	main()


	