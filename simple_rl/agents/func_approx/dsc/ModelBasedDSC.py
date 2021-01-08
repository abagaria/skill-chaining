import ipdb
import itertools
from copy import deepcopy
from simple_rl.agents.func_approx.dsc.MBOptionClass import ModelBasedOption
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain


class ModelBasedSkillChaining(object):
    def __init__(self, mdp, max_steps,
                 use_vf, use_model,
                 use_dense_rewards,
                 experiment_name, device,
                 gestation_period, buffer_length,
                 generate_init_gif,
                 seed, multithread_mpc):

        self.mdp = mdp

        self.device = device
        self.use_vf = use_vf
        self.use_model = use_model
        self.experiment_name = experiment_name
        self.max_steps = max_steps
        self.use_dense_rewards = use_dense_rewards
        self.multithread_mpc = multithread_mpc

        self.lr_a = 3e-4
        self.lr_c = 3e-4

        self.seed = seed
        self.generate_init_gif = generate_init_gif

        self.buffer_length = buffer_length
        self.gestation_period = gestation_period

        self.global_option = self.create_global_model_based_option()

        self.chains = []
        self.new_options = []
        self.mature_options = []
        self.current_option_idx = 1

    # ------------------------------------------------------------
    # Act functions
    # ------------------------------------------------------------

    @staticmethod
    def _pick_earliest_option(state, options):
        for option in options:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option

    def _pick_chain_corresponding_to_goal(self, goal):
        for chain in self.chains:
            if chain.target_salient_event(goal):
                return chain

    def _pick_chain_corresponding_to_goal_2(self, goal):
        for chain in self.chains:
            for option in chain.options:
                if option.is_in_effect_set(goal):
                    return chain

    def act(self, state, goal):
        chain = self._pick_chain_corresponding_to_goal(goal)
        if chain is None:
            chain = self._pick_chain_corresponding_to_goal_2(goal)
        if chain is not None:
            for option in chain.options:
                if option.is_init_true(state):
                    subgoal = option.get_goal_for_rollout()
                    if not option.is_at_local_goal(state, subgoal):
                        return option, subgoal
        return self.global_option, goal

    # ------------------------------------------------------------
    # Option off-policy termination triggers
    # ------------------------------------------------------------

    def _get_current_salient_events(self, state):
        """ Get all the salient events satisfied by the current `state`. """
        events = self.mdp.get_all_target_events_ever() + [self.mdp.get_start_state_salient_event()]
        current_events = [event for event in events if event(state)]
        return current_events

    def disable_triggering_options_targeting_init_event(self, state):
        """ Disable off-policy triggers for all options that target the current salient events. """
        current_salient_events = self._get_current_salient_events(state)
        trained_options = itertools.chain.from_iterable([chain.options for chain in self.chains])

        for current_salient_event in current_salient_events:
            for option in trained_options:
                if option.chain_id is not None:
                    option.started_at_target_salient_event = option.target_salient_event == current_salient_event

        # Additionally, disable off-policy triggering options whose termination set we are in
        for option in trained_options:
            option.started_at_target_salient_event = option.is_term_true(state)

    def trigger_option_termination_conditions_off_policy(self,
                                                         untrained_option,
                                                         executed_option,
                                                         option_transitions,
                                                         episode_number):
        if untrained_option != executed_option:
            if untrained_option.trigger_termination_condition_off_policy(option_transitions, episode_number) \
                    and untrained_option.get_training_phase() == "initiation_done":
                print(f"Triggered final termination condition for {untrained_option} while executing {executed_option}")

    # ------------------------------------------------------------
    # Managing the skill chains
    # ------------------------------------------------------------

    def should_create_children_options(self, parent_option):

        if parent_option is None:
            return False

        return self.should_create_more_options() \
               and parent_option.get_training_phase() == "initiation_done" \
               and self.chains[parent_option.chain_id - 1].should_continue_chaining()

    def should_create_more_options(self):
        """ Continue chaining as long as any chain is still accepting new options. """
        return any([chain.should_continue_chaining() for chain in self.chains])

    def add_new_option_to_skill_chain(self, new_option):
        self.new_options.append(new_option)
        chain_idx = new_option.chain_id - 1
        self.chains[chain_idx].options.append(new_option)

    def finish_option_initiation_phase(self, untrained_option):
        if untrained_option.get_training_phase() == "initiation_done":
            self.new_options.remove(untrained_option)
            self.mature_options.append(untrained_option)
            return untrained_option

    def manage_chain_after_option_rollout(self, *,
                                          executed_option,
                                          created_options,
                                          option_transitions, episode):
        """" After a single option rollout, this method will perform the book-keeping necessary to maintain skill-chain. """

        for option in self.new_options:
            self.trigger_option_termination_conditions_off_policy(option,
                                                                  executed_option,
                                                                  option_transitions,
                                                                  episode_number=episode)

            completed_option = self.finish_option_initiation_phase(untrained_option=option)
            should_create_children = self.should_create_children_options(completed_option)
            if should_create_children:
                new_option = self.create_child_option(completed_option)
                self.add_new_option_to_skill_chain(new_option)

            if completed_option is not None:
                created_options.append(completed_option)

        return created_options

    # ------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------

    def dsc_rollout(self, episode_number, num_steps, goal, interrupt_handle=lambda x: False):
        step_number = 0
        created_options = []

        while step_number < num_steps and not interrupt_handle(self.mdp.cur_state):
            state = deepcopy(self.mdp.cur_state)
            selected_option, subgoal = self.act(state, goal)

            self.disable_triggering_options_targeting_init_event(state)
            transitions, reward = selected_option.rollout(episode=episode_number,
                                                          step_number=step_number,
                                                          rollout_goal=subgoal)

            if len(transitions) == 0:
                break

            created_options = self.manage_chain_after_option_rollout(executed_option=selected_option,
                                                                     created_options=created_options,
                                                                     option_transitions=transitions,
                                                                     episode=episode_number)
            step_number += len(transitions)

        return step_number, created_options

    # ------------------------------------------------------------
    # Convenience functions
    # ------------------------------------------------------------

    def get_all_options(self):
        return itertools.chain.from_iterable([chain.options for chain in self.chains])

    def create_new_chain(self, *, init_event, target_event):
        chain_id = len(self.chains) + 1
        name = f"goal-option-{chain_id}"
        root_option = self.create_model_based_option(name=name,
                                                     parent=None,
                                                     chain_id=chain_id,
                                                     init_event=init_event,
                                                     target_event=target_event)
        self.new_options.append(root_option)

        new_chain = SkillChain(options=[root_option], chain_id=chain_id,
							   target_salient_event=target_event,
							   init_salient_event=init_event,
							   mdp_init_salient_event=self.mdp.get_start_state_salient_event())
        self.add_skill_chain(new_chain)
        return new_chain

    def add_skill_chain(self, new_chain):
        if new_chain not in self.chains:
            self.chains.append(new_chain)

    def create_child_option(self, parent_option):

        # Create new option whose termination is the initiation of the option we just trained
        if "goal_option" in parent_option.name:
            prefix = f"option_{parent_option.name.split('_')[-1]}"
        else:
            prefix = parent_option.name
        name = prefix + "_{}".format(len(parent_option.children))
        print("Creating {} with parent {}".format(name, parent_option.name))

        new_option = self.create_model_based_option(name=name, parent=parent_option,
                                                    chain_id=parent_option.chain_id,
                                                    init_event=parent_option.init_salient_event,
                                                    target_event=parent_option.target_salient_event)
        parent_option.children.append(new_option)
        return new_option

    def create_model_based_option(self, *, name, parent, chain_id, init_event, target_event):
        assert chain_id is not None, chain_id
        assert target_event is not None, target_event

        option = ModelBasedOption(parent=parent, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=False,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=self.max_steps, device=self.device,
                                  init_salient_event=init_event,
                                  target_salient_event=target_event,
                                  name=name,
                                  path_to_model="",
                                  global_solver=self.global_option.solver,
                                  use_vf=self.use_vf,
                                  use_global_vf=self.use_vf,
                                  use_model=self.use_model,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=self.global_option.value_learner,
                                  option_idx=self.current_option_idx,
                                  multithread_mpc=self.multithread_mpc,
                                  chain_id=chain_id)
        return option

    def create_global_model_based_option(self):
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=True,
                                  gestation_period=self.gestation_period,
                                  timeout=100, max_steps=self.max_steps, device=self.device,
                                  target_salient_event=None,
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=self.use_vf,
                                  use_global_vf=self.use_vf,
                                  use_model=self.use_model,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=None,
                                  option_idx=0,
                                  lr_c=self.lr_c, lr_a=self.lr_a,
                                  multithread_mpc=self.multithread_mpc,
                                  chain_id=0)
        return option
