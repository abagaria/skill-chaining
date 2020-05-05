import pickle
import numpy as np

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



def main():
	test_data = load_data('runs/(cs2951x) maze_old_fixed_learning/run_1_all_data/num_options_history.pkl')
	last_option, idx = 0, None
	for i,  option in enumerate(test_data):
		if option > last_option:
			last_option, idx = option, i
	
	print(last_option, idx)
	# test_data = load_data('runs/(cs2951x) maze_chain_break_nu_0.6/run_1_all_data/num_options_history.pkl')
	# print(test_data)

	# test_obj = ClassTest(1, [1,2,3])
	# get_skill_chain()

	# print(np.linspace(34, 40, 6, dtype=int))

	if True:
		pass
	else:
		print(False)


if __name__ == "__main__":
	main()


	