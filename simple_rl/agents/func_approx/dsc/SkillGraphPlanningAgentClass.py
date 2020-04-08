from tqdm import tqdm
from collections import defaultdict
from simple_rl.mdp.MDPClass import MDP
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining


class SkillGraphPlanningAgent(object):
    def __init__(self, mdp, chainer, include_salient_events=False):
        """

        Args:
            mdp (MDP)
            chainer (SkillChaining)
            include_salient_events (bool): Whether to include SalientEvent objects in the graph
        """
        self.mdp = mdp
        self.chainer = chainer
        self.include_salient_events = include_salient_events

        self.visitation_counts = defaultdict(lambda : defaultdict(lambda : 0))

    def construct_abstract_mdp_from_executions(self, num_rollouts):

        def _get_executed_option(trajectories):
            trajectory = trajectories[-1]
            option_and_state = trajectory[0]
            option_idx = option_and_state[0]
            option = self.chainer.trained_options[option_idx]  # type: Option
            return option

        def _get_resulting_options(trajectories):
            trajectory = trajectories[-1]
            option_and_state = trajectory[0]
            resulting_state = option_and_state[1]

            # Get all the options for which I(s') is true and beta(s') is false
            next_option_events = [option.is_init_true for option in self.chainer.trained_options[1:]
                                  if option.is_init_true(resulting_state) and not option.is_term_true(resulting_state)]

            # Get all the salient events which are satisfied at s'
            if self.include_salient_events:
                next_option_events += [salient_event for salient_event in self.mdp.get_original_salient_events() if salient_event(resulting_state)]

            # Combine and return all the predicate functions satisfied at state s'
            return next_option_events

        summed_reward = 0.

        for _ in tqdm(range(num_rollouts), desc="Constructing abstract MDP"):
            r, option_trajectories = self.chainer.trained_forward_pass(render=False)
            if len(option_trajectories) > 0:
                executed_option = _get_executed_option(option_trajectories)
                resulting_options = _get_resulting_options(option_trajectories)
                for resulting_option in resulting_options:
                    self.visitation_counts[executed_option][resulting_option] += 1
            summed_reward += r

        return summed_reward

    def construct_abstract_mdp_from_connections(self):
        pass
