import pickle

def load_data(file_name):
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data

class ClassTest(object):

	def __init__(self, episode):
		self.episode = episode
	
	def update_episode(self, episode):
		self.episode = episode

def main():
	# test_data = load_data('runs/(cs2951x) maze_chain_break_nu_0.6/run_1_all_data/options_chain_breaks.pkl')
	# print(set(test_data['option_1']))

	test_data = load_data('runs/(cs2951x) maze_chain_break_nu_0.6/run_1_all_data/num_options_history.pkl')
	print(test_data)


if __name__ == "__main__":
	main()


	