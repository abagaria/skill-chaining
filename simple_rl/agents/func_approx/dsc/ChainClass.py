import ipdb
import numpy as np
from simple_rl.agents.func_approx.dsc.MBOptionClass import ModelBasedOption
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.mdp.StateClass import State


class SkillChain(object):
    def __init__(self, init_salient_event, target_salient_event, options, chain_id, mdp_init_salient_event):
        """
        Data structure that keeps track of all options in a particular chain,
        where each chain is identified by a unique target salient event. Chain here
        may also refer to Skill Trees.
        Args:
            init_salient_event (SalientEvent): f: s -> {0, 1} based on start salience
            target_salient_event (SalientEvent): f: s -> {0, 1} based on target salience
            options (list): list of options in the current chain
            chain_id (int): Identifier for the current skill chain
            mdp_init_salient_event (SalientEvent): Start state salient event of the overall MDP
        """
        self.options = options
        self.init_salient_event = init_salient_event
        self.target_salient_event = target_salient_event
        self.mdp_init_salient_event = mdp_init_salient_event
        self.chain_id = chain_id

        # Data structures for determining when a skill chain is completed
        self._init_descendants = []
        self._init_ancestors = []
        self._is_deemed_completed = False
        self.completing_vertex = None

        self.max_num_options = 4

    def __eq__(self, other):
        return self.chain_id == other.chain_id

    def __str__(self):
        return "SkillChain-{}".format(self.chain_id)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.options)

    def __getitem__(self, item):
        return self.options[item]

    def set_chain_completed(self):
        self._is_deemed_completed = True

    def set_init_descendants(self, descendants):
        """ The `descendants` are the set of vertices that you can get to from the chain's init-salient-event. """
        if len(descendants) > 0:
            assert all([isinstance(node, (ModelBasedOption, SalientEvent)) for node in descendants]), f"{descendants}"
            self._init_descendants = descendants

    def set_init_ancestors(self, ancestors):
        """ The `ancestors` are the set of vertices from which you can get to the chain's init-salient-event. """
        if len(ancestors) > 0:
            assert all([isinstance(node, (ModelBasedOption, SalientEvent)) for node in ancestors]), f"{ancestors}"
            self._init_ancestors = ancestors

    def should_continue_chaining(self):
        return not self.is_chain_completed()

    @staticmethod
    def should_exist_edge_between_options(my_option, other_option):
        """ Should there exist an edge from option1 -> option2? """
        assert isinstance(my_option, ModelBasedOption)
        assert isinstance(other_option, ModelBasedOption)

        if my_option.get_training_phase() == "initiation_done" and other_option.get_training_phase() == "initiation_done":
            effect_set_matrix = my_option.get_effective_effect_set()
            if len(effect_set_matrix) > 0:
                inits = other_option.pessimistic_batched_is_init_true(effect_set_matrix)
                is_intersecting = inits.all()
                return is_intersecting
        return False

    @staticmethod
    def should_exist_edge_from_event_to_option(event, option):
        """ Should there be an edge from `event` to `option`? """
        assert isinstance(event, SalientEvent)
        assert isinstance(option, ModelBasedOption)

        if option.get_training_phase() == "initiation_done":
            if len(event.trigger_points) > 0:  # Be careful: all([]) = True
                inits = option.pessimistic_batched_is_init_true(event.trigger_points)
                is_intersecting = inits.all()
                return is_intersecting
            return option.is_init_true(event.target_state)
        return False

    @staticmethod
    def should_exist_edge_from_option_to_event(option, event):
        """ Should there be an edge from `option` to `event`? """
        assert isinstance(option, ModelBasedOption)
        assert isinstance(event, SalientEvent)

        if option.get_training_phase() == "initiation_done":
            effect_set = option.effect_set  # list of states
            inits = event.batched_is_init_true(effect_set)
            return inits.all()
        return False

    def should_expand_initiation_classifier(self, option):
        assert isinstance(option, ModelBasedOption), f"{type(option)}"

        # Hack: For the first skill-chain, we limit the number of options to 4
        trained_options = [o for o in self.options if o.get_training_phase() == 'initiation_done']
        if self.chain_id == 1 and len(trained_options) > self.max_num_options:
            return True

        if len(self.init_salient_event.trigger_points) > 0:
            return any([option.is_init_true(s) for s in self.init_salient_event.trigger_points])
        if self.init_salient_event.get_target_position() is not None:
            return option.is_init_true(self.init_salient_event.get_target_position())
        
        return False

    def should_complete_chain(self, option):
        """ Check if a newly learned option completes its corresponding chain. """
        assert isinstance(option, ModelBasedOption), f"{type(option)}"

        # If there is a path from a descendant of the chain's init salient event
        # to the newly learned option, then that chain's job is done
        for descendant in self._init_descendants:
            if isinstance(descendant, SalientEvent):
                if self.should_exist_edge_from_event_to_option(descendant, option):
                    self.completing_vertex = descendant, "descendant"
                    return True
            if isinstance(descendant, ModelBasedOption):
                if self.should_exist_edge_between_options(descendant, option):
                    self.completing_vertex = descendant, "descendant"
                    return True

        # If not, then check if there is a direct path from the init-salient-event to the new option
        if self.should_exist_edge_from_event_to_option(self.init_salient_event, option):
            self.completing_vertex = self.init_salient_event, "init_event"
            return True

        # TODO: Can't check for intersection with start event for backward chains - can we cap by num_skills?
        trained_options = [option for option in self.options if option.get_training_phase() == 'initiation_done']
        return len(trained_options) > self.max_num_options

    def is_chain_completed(self):
        """
        The chain is considered complete when it learns an option whose initiation set covers
        at least one of the descendants of the chain's init-salient-event.

        Returns:
            is_completed (bool)
        """
        if self._is_deemed_completed:
            return True

        completed_options = [option for option in self.options if option.get_training_phase() == "initiation_done"]

        for option in completed_options:
            if self.should_expand_initiation_classifier(option):
                option.expand_initiation_classifier(self.init_salient_event)
                if self.should_complete_chain(option):
                    self._is_deemed_completed = True
                    return True

        for option in completed_options:
            if self.should_complete_chain(option):
                self._is_deemed_completed = True
                return True

        return False

    def get_leaf_nodes_from_skill_chain(self):
        return [option for option in self.options if len(option.children) == 0]

    def get_root_nodes_from_skill_chain(self):
        return [option for option in self.options if option.parent is None]

    @staticmethod
    def _get_position(state):
        return GymMDP.get_position(state)
