import ipdb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from simple_rl.agents.func_approx.ppo_rnd.model import CnnActorCriticNetwork, RNDModel
from simple_rl.agents.func_approx.ppo_rnd.replay_buffer import ReplayMemory
from simple_rl.agents.func_approx.ppo_rnd.utils import global_grad_norm_, RunningMeanStd


class RNDAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.001,
            clip_grad_norm=0.5,
            epoch=4,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=1.0,
            use_cuda=False,
            use_noisy_net=False):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.output_size = output_size
        self.input_size = input_size
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.int_coef = 1.
        self.ext_coef = 2.
        self.update_interval = 2048

        self.rnd = RNDModel(input_size, output_size)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()),
                                    lr=learning_rate)

        self.rnd = self.rnd.to(self.device)
        self.model = self.model.to(self.device)

        self.memory = ReplayMemory()
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(1, 84, 84))

    def act(self, state):  # TODO: This might not work with single states
        state = state[None, ...]
        state = torch.Tensor(state).to(self.device).float()

        with torch.no_grad():
            policy, value_ext, value_int = self.model(state)
            action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)

        return action[0], \
               value_ext.data.item(),\
               value_int.data.item(),\
               policy.detach()

    def step(self, *, state, action, policy, ext_reward, int_reward, ext_value, int_value, next_state, done):
        self.memory.store(state, action, policy, ext_reward, int_reward, ext_value, int_value, next_state, done)

        if len(self.memory) >= self.update_interval:
            self.learn()
            self.memory.clear()

    def learn(self):
        state, action, policy, reward, int_reward, ext_value, int_value, next_state, next_obs, done = self.memory.retrieve()

        self.update_reward_rms(int_reward)
        int_reward = self.normalize_intrinsic_rewards(int_reward)

        print(f"Mean V_int: {int_value.mean()} | Max V_int: {int_value.max()} | Mean V_ext: {ext_value.mean()} | Max V_ext: {ext_value.max()}")

        ext_target, ext_adv = self.make_train_data(reward,
                                                   done,
                                                   ext_value,
                                                   self.gamma)

        int_target, int_adv = self.make_train_data(int_reward,
                                                   np.zeros_like(int_reward),
                                                   int_value,
                                                   self.gamma)

        total_adv = (int_adv * self.int_coef) + (ext_adv * self.ext_coef)

        self.obs_rms.update(next_obs)

        normalized_next_obs = ((next_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5)

        self.train_model(np.float32(state) / 255., ext_target, int_target, action,
                          total_adv, normalized_next_obs, policy)

    def make_train_data(self, rewards, dones, values, gamma):
        assert isinstance(values, np.ndarray), values
        assert rewards.shape == values.shape, f"{rewards.shape, values.shape}"

        discounted_returns = self.get_n_step_returns(rewards, dones, gamma)
        advantages = discounted_returns - values
        return discounted_returns, advantages

    def get_n_step_returns(self, undiscounted_rewards, dones, gamma):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(undiscounted_rewards), reversed(dones)):
            discounted_reward = 0. if done else discounted_reward
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        discounted_returns = np.array(rewards)
        return discounted_returns

    def update_reward_rms(self, episodic_rewards):
        """ Compute the mean, std and len of the rewards and update the reward_rms with it. """
        num_steps = len(episodic_rewards)
        gammas = np.power(self.gamma * np.ones(num_steps), np.arange(0, num_steps))
        discounted_rewards = gammas * episodic_rewards

        assert len(discounted_rewards) == len(episodic_rewards)

        if len(discounted_rewards) > 0:
            mean = np.mean(discounted_rewards)
            std = np.std(discounted_rewards)
            size = len(discounted_rewards)

            self.reward_rms.update_from_moments(mean, std ** 2, size)

    def normalize_intrinsic_rewards(self, intrinsic_rewards):
        return intrinsic_rewards / np.sqrt(self.reward_rms.var)

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, next_obs):
        assert next_obs.max() < +5.1, next_obs.max()
        assert next_obs.min() > -5.1, next_obs.min()

        next_obs = torch.FloatTensor(next_obs).to(self.device)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.item()

    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        print(f"[PPO] Mean intrinsic target: {target_int_batch.mean()} | Extrinsic target: {target_ext_batch.mean()}")

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx].unsqueeze(1))

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # ---------------------------------------------------------------------------------

                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss
                loss.backward()
                global_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor.parameters()))
                self.optimizer.step()


def pre_train(agent, env, num_steps):
    next_obs = []
    env.reset()

    for step in range(num_steps):
        action = env.action_space.sample()

        s, r, d, _ = env.step(action)
        next_obs.append(s[-1, ...].reshape([1, 84, 84]))

        if d: env.reset()

    next_obs = np.array(next_obs).squeeze()
    agent.obs_rms.update(next_obs)


def train(agent, env, num_steps):
    assert isinstance(agent, RNDAgent)
    state = env.reset()

    score = 0.
    intrinsic_score = 0.
    cumulative_score = 0.
    steps_per_episode = 0
    episodic_rewards = []

    for step in range(num_steps):
        action, value_ext, value_int, policy = agent.act(np.float32(state) / 255.)
        next_state, reward, done, info = env.step(action)

        next_obs = next_state[-1, ...][None, ...][None, ...]
        normalized_next_obs = ((next_obs - agent.obs_rms.mean) / agent.obs_rms.var).clip(-5, 5)
        int_reward = agent.compute_intrinsic_reward(normalized_next_obs)

        agent.step(state=state, action=action, policy=policy,
                   ext_reward=reward, int_reward=int_reward, ext_value=value_ext,
                   int_value=value_int, next_state=next_state, done=done)

        state = next_state
        score += reward
        cumulative_score += reward
        intrinsic_score += int_reward
        steps_per_episode += 1

        if done or steps_per_episode > 4500:
            episodic_rewards.append(score)
            print(f"Step: {step} | Score: {score} | Cum Score: {cumulative_score} | Intrinsic Reward: {intrinsic_score} | Var: {agent.reward_rms.var}")
            score = 0.
            intrinsic_score = 0.
            steps_per_episode = 0
            state = env.reset()

    return episodic_rewards


if __name__ == "__main__":
    from simple_rl.agents.func_approx.ppo_rnd.montezuma import get_montezuma_env
    env = get_montezuma_env()
    rnd = RNDAgent(
        input_size=env.observation_space.shape,
        output_size=env.action_space.n,
        gamma=0.99,
        use_cuda=False,
    )
    pre_train(rnd, env, num_steps=1000)
    er = train(rnd, env, num_steps=int(1e7))
