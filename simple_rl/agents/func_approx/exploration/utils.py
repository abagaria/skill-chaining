import matplotlib.pyplot as plt
import seaborn as sns


def visualize_bonus_for_actions(density_model, options, episode, experiment_name, seed):
    for option in options:
        visualize_bonus_for_action(density_model, option.option_idx, episode, experiment_name, seed, option.name)

def visualize_bonus_for_action(density_model, action, episode, experiment_name, seed, action_name=""):
    sns.set_style("white")
    s_a_bonus = density_model.s_a_bonus
    x, y, z = [], [], []

    for state in s_a_bonus:
        if action in s_a_bonus[state]:
            x.append(state[0])
            y.append(state[1])
            z.append(s_a_bonus[state][action])

    plt.scatter(x, y, None, c=z)
    plt.colorbar()
    plt.title("Bonus @ Episode # {} for action {}".format(episode, action))
    plt.savefig("kde_plots/{}/bonus_action_{}_episode_{}_seed_{}.png".format(experiment_name,
                                                                             str(action) + action_name,
                                                                             episode, seed))
    plt.close()
