import ipdb
import pfrl
import argparse
from pfrl.agents import PPO
from copy import deepcopy
from simple_rl.agents.func_approx.td3.utils import *
from simple_rl.agents.func_approx.ppo.model import fc_model
from simple_rl.agents.func_approx.rnd.RNDRewardLearner import RND
from simple_rl.agents.func_approx.td3.replay_buffer import ReplayBuffer
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP


class PPOAgent(object):
    def __init__(self,
                 obs_size,
                 action_size,
                 lr=3e-4,
                 update_interval=2048,
                 batchsize=64,
                 epochs=4,
                 device_id=0,
                 maintain_replay_buffer=False,
                 use_rnd_for_exploration=False):

        self.device_id = device_id
        self.maintain_replay_buffer = maintain_replay_buffer
        self.use_rnd_for_exploration = use_rnd_for_exploration
        self.device = torch.device(f"cuda:{device_id}" if device_id > -1 else "cpu")

        self.model = fc_model(obs_size, action_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        # Normalize observations based on their empirical mean and variance
        self.obs_normalizer = pfrl.nn.EmpiricalNormalization(
            obs_size, clip_threshold=40.  # Not using a clip threshold because the maze is larger than it
        )

        self.agent = PPO(self.model,
                         optimizer,
                         gpu=device_id,
                         obs_normalizer=self.obs_normalizer,
                         update_interval=update_interval,
                         minibatch_size=batchsize,
                         epochs=epochs,
                         clip_eps_vf=None,
                         entropy_coef=1e-3,  # Taken from RND paper (in pfrl, this is 0)
                         standardize_advantages=True,
                         gamma=0.995,
                         lambd=0.97
        )

        if use_rnd_for_exploration:
            self.rnd = RND(obs_size,
                           lr=1e-4,
                           n_epochs=1,
                           batch_size=64,
                           device=self.device,
                           update_interval=2048,
                           use_reward_norm=True)

        # Maintaining a replay buffer for visualizations -- not used by the PPO agent
        if maintain_replay_buffer:
            self.replay_buffer = ReplayBuffer(obs_size, action_size, device=torch.device("cpu"))

    def act(self, obs):
        return self.agent.act(obs)

    def step(self, obs, action, reward, next_obs, done):
        self.agent.observe(next_obs, reward, done, reset=False)

        if self.use_rnd_for_exploration:
            self.rnd.update(next_obs)

        if self.maintain_replay_buffer:
            self.replay_buffer.add(obs, action, reward, next_obs, done)

    def get_batched_qvalues(self, states):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        device = f"cuda:{self.device_id}" if self.device_id >= 0 else "cpu"
        states = states.to(device)
        with torch.no_grad():
            action_values = self.agent.model(states)
        return action_values.q_values

    def value_function(self, states):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        device = f"cuda:{self.device_id}" if self.device_id >= 0 else "cpu"
        states = states.to(device)

        with torch.no_grad(), pfrl.utils.evaluating(self.agent.model):
            states = self.obs_normalizer(states, update=False)
            _, values = self.agent.model(states)

        values = values.squeeze() if len(states) > 1 else values
        return values.cpu().numpy()


def make_chunked_vf_plot(solver, experiment_name, episode):
    states = np.array([exp[0] for exp in solver.replay_buffer])
    values = chunked_inference(solver.value_function, states, device=solver.device)
    plt.scatter(states[:, 0], states[:, 1], c=values)
    plt.colorbar()
    plt.savefig(f"value_function_plots/{experiment_name}/vf_episode_{episode}.png")
    plt.close()
    return values.max()


def train(agent, mdp, episodes, steps, experiment_name, starting_episode=0):

    def _get_intrinsic_reward(s):
        return agent.rnd.get_reward(s.features()[None, ...]).detach().cpu().numpy()[0]

    def _features(s):
        return s.features().astype(np.float32, copy=False)

    per_episode_scores = []
    per_episode_durations = []
    per_episode_intrinsic_rewards = []

    visited_x_positions = []
    visited_y_positions = []

    for episode in range(starting_episode, starting_episode + episodes):
        mdp.reset()
        state = deepcopy(mdp.cur_state)

        score = 0.
        intrinsic_score = 0.
        episodic_rewards = []

        for step in range(steps):

            action = agent.act(_features(state))
            reward, next_state = mdp.execute_agent_action(action)
            intrinsic_reward = _get_intrinsic_reward(next_state)

            agent.step(_features(state), action, intrinsic_reward, _features(next_state), next_state.is_terminal())

            score += reward
            state = next_state
            intrinsic_score += intrinsic_reward
            episodic_rewards.append(intrinsic_score)

            visited_x_positions.append(next_state.position[0])
            visited_y_positions.append(next_state.position[1])

            if state.is_terminal():
                break

        if agent.rnd.use_reward_norm:
            agent.rnd.update_reward_rms(episodic_rewards)

        final_state = np.round(_features(next_state)[:2], decimals=2)
        per_episode_scores.append(score)
        per_episode_durations.append(step)
        per_episode_intrinsic_rewards.append(intrinsic_score)

        print(f"Episode: {episode} | Intrinsic Reward: {intrinsic_score} | Var: {agent.rnd.reward_rms.var} | Final State: {final_state}")

        if episode % 100 == 0:
            if hasattr(agent, "replay_buffer"):
                max_int = make_chunked_vf_plot(agent, experiment_name, episode)
                print(f"Max intrinsic value: {max_int}")

            plt.figure()
            plt.scatter(visited_x_positions, visited_y_positions, alpha=0.3)
            plt.title(f"Episode {episode}")
            plt.savefig(f"value_function_plots/{experiment_name}/visited_states_episode_{episode}.png")
            plt.close()

    return per_episode_scores, per_episode_durations, per_episode_intrinsic_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--render", type=bool, help="render environment training", default=False)
    parser.add_argument("--episodes", type=int, help="number of training episodes", default=200)
    parser.add_argument("--steps", type=int, help="number of steps per episode", default=200)
    parser.add_argument("--device", type=str, help="cuda/cpu", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log_dir = create_log_dir(args.experiment_name)
    create_log_dir("saved_runs")
    create_log_dir("value_function_plots")
    create_log_dir("initiation_set_plots")
    create_log_dir("value_function_plots/{}".format(args.experiment_name))
    create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

    mdp = D4RLAntMazeMDP(seed=args.seed,
                         goal_state=None,
                         maze_size="umaze",
                         render=args.render,
                         dense_reward=False)

    agent = PPOAgent(obs_size=mdp.state_space_size(),
                     action_size=mdp.action_space_size(),
                     maintain_replay_buffer=True,
                     lr=3e-4, use_rnd_for_exploration=True)

    pes, ped, pei = train(agent, mdp, args.episodes, args.steps, args.experiment_name)
