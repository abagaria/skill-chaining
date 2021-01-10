def h(self, episode, name):
    pkl = {"vf": {}, "init":{}}
    options = self.mature_options + self.new_options
    for option in self.mature_options:
        episode_label = episode if self.generate_init_gif else -1
        pkl['init'][option.name] = pkl_plot_two_class_classifier(option, episode_label, self.experiment_name, plot_examples=True)
    for option in options:
        if self.use_global_vf:
            pkl['vf'][option.name] = pkl_make_chunked_goal_conditioned_value_function_plot(option.global_value_learner,
                                                            goal=option.get_goal_for_rollout(),
                                                            episode=episode, seed=self.seed,
                                                            experiment_name=self.experiment_name,
                                                            option_idx=option.option_idx)
    pickle.dump(pkl, open(f"{name}.pkl", "wb"))

def pkl_plot_one_class_initiation_classifier(option):
    colors = ["blue", "yellow", "green", "red", "cyan", "brown"]

    X = option.construct_feature_matrix(option.positive_examples)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z1 = option.pessimistic_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    color = colors[option.option_idx % len(colors)]
    return xx, yy, Z1
    # plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors=[color])

def pkl_plot_two_class_classifier(option, episode, experiment_name, plot_examples=True, seed=0):
    states = get_grid_states(option.overall_mdp)
    values = get_initiation_set_values(option)

    x = np.array([state[0] for state in states])
    y = np.array([state[1] for state in states])
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xx, yy = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    zz = rbf(xx, yy)
    # plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", alpha=0.6, cmap=plt.cm.coolwarm)
    # plt.colorbar()

    # Plot trajectories
    positive_examples = option.construct_feature_matrix(option.positive_examples)
    negative_examples = option.construct_feature_matrix(option.negative_examples)

    if positive_examples.shape[0] > 0 and plot_examples:
        pass
        # plt.scatter(positive_examples[:, 0], positive_examples[:, 1], label="positive", c="black", alpha=0.3, s=10)

    if negative_examples.shape[0] > 0 and plot_examples:
        pass
        # plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", c="lime", alpha=1.0, s=10)

    if option.pessimistic_classifier is not None:
        p_xx, p_yy, p_Z1 = pkl_plot_one_class_initiation_classifier(option)

    # background_image = imageio.imread("four_room_domain.png")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

    name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)

    return (x, y, xi, yi, xx, yy, zz, p_xx, p_yy, p_Z1)
    # plt.title("{} Initiation Set".format(option.name))
    # plt.savefig("initiation_set_plots/{}/{}_initiation_classifier_{}.png".format(experiment_name, name, seed))
    # plt.close()

def pkl_make_chunked_goal_conditioned_value_function_plot(solver, goal, episode, seed, experiment_name, chunk_size=1000, replay_buffer=None, option_idx=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer

    # Take out the original goal and append the new goal
    states = [exp[0] for exp in replay_buffer]
    states = [state[:-2] for state in states]
    states = np.array([np.concatenate((state, goal), axis=0) for state in states])

    actions = np.array([exp[1] for exp in replay_buffer])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for chunk_number, (state_chunk, action_chunk) in tqdm(enumerate(zip(state_chunks, action_chunks)), desc="Making VF plot"):  # type: (int, np.ndarray)
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        if isinstance(solver, DQNAgent):
            chunk_qvalues = solver.get_batched_qvalues(state_chunk).cpu().numpy()
            chunk_qvalues = np.max(chunk_qvalues, axis=1)
        else:
            chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    # plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    # plt.colorbar()

    if option_idx is None:
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    else:
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}_option_{option_idx}"
    # plt.title(f"VF Targeting {np.round(goal, 2)}")
    # plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    # plt.close()

    return (states, qvalues, goal)