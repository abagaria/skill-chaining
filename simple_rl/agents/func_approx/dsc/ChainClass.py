import ipdb
import numpy as np
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.mdp.StateClass import State


class SkillChain(object):
    def __init__(self, init_salient_event, target_salient_event, options, chain_id, mdp_init_salient_event,
                 intersecting_options=[], option_intersection_salience=False, event_intersection_salience=True):
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
            intersecting_options (list): List of options whose initiation sets overlap
            option_intersection_salience (bool): Whether to chain until the current chain intersects with another option
            event_intersection_salience (bool): Chain until you intersect with another salient event
        """
        self.options = options
        self.init_salient_event = init_salient_event
        self.target_salient_event = target_salient_event
        self.mdp_init_salient_event = mdp_init_salient_event
        self.chain_id = chain_id
        self.intersecting_options = intersecting_options

        self.option_intersection_salience = option_intersection_salience
        self.event_intersection_salience = event_intersection_salience

        # Data structures for determining when a skill chain is completed
        self._init_descendants = []
        self._init_ancestors = []
        self._is_deemed_completed = False
        self.completing_vertex = None

        if target_salient_event is None and len(intersecting_options) > 0:
            self.target_predicate = lambda s: all([option.is_init_true(s) for option in intersecting_options])

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
            assert all([isinstance(node, (Option, SalientEvent)) for node in descendants]), f"{descendants}"
            self._init_descendants = descendants

    def set_init_ancestors(self, ancestors):
        """ The `ancestors` are the set of vertices from which you can get to the chain's init-salient-event. """
        if len(ancestors) > 0:
            assert all([isinstance(node, (Option, SalientEvent)) for node in ancestors]), f"{ancestors}"
            self._init_ancestors = ancestors

    def state_in_chain(self, state):
        """ Is state inside the initiation set of any of the options in the chain. """

        # We consider the state is in the target salient event, it can be
        # considered to be inside the skill-chain
        if self.target_salient_event(state):
            return True

        for option in self.options:  # type: Option
            if option.initiation_classifier is not None and \
                    option.is_init_true(state) and option.get_training_phase() == "initiation_done":
                return True
        return False

    def get_option_for_state(self, state):

        if self.target_salient_event(state) and len(self.options) > 0:
            goal_option = self.options[0]
            assert goal_option.parent is None
            assert goal_option.is_term_true(state)
            return goal_option

        for option in self.options:  # type: Option
            if option.initiation_classifier is not None and \
                    option.is_init_true(state):
                return option
        return None

    def should_continue_chaining(self):
        return not self.is_chain_completed()

    @staticmethod
    def should_exist_edge_between_options(my_option, other_option):
        """ Should there exist an edge from option1 -> option2? """
        if my_option.get_training_phase() == "initiation_done" and other_option.get_training_phase() == "initiation_done":
            effect_set = my_option.effect_set  # list of states
            effect_set_matrix = SkillChain.get_position_matrix(effect_set)
            inits = other_option.batched_is_init_true(effect_set_matrix)
            is_intersecting = inits.all()
            return is_intersecting
        return False

    @staticmethod
    def should_exist_edge_from_event_to_option(event, option):
        """ Should there be an edge from `event` to `option`? """
        if option.get_training_phase() == "initiation_done":
            if len(event.trigger_points) > 0:  # Be careful: all([]) = True
                state_matrix = SkillChain.get_position_matrix(event.trigger_points)
                inits = option.batched_is_init_true(state_matrix)
                is_intersecting = inits.all()
                return is_intersecting
            return option.is_init_true(event.target_state)
        return False

    @staticmethod
    def should_exist_edge_from_option_to_event(option, event):
        """ Should there be an edge from `option` to `event`? """
        if option.get_training_phase() == "initiation_done":
            effect_set = option.effect_set  # list of states
            effect_set_matrix = SkillChain.get_position_matrix(effect_set)
            inits = event.batched_is_init_true(effect_set_matrix)
            return inits.all()
        return False

    def should_expand_initiation_classifier(self, option):
        assert isinstance(option, Option), f"{type(option)}"

        if len(self.init_salient_event.trigger_points) > 0:
            return any([option.is_init_true(s) for s in self.init_salient_event.trigger_points])
        if self.init_salient_event.get_target_position() is not None:
            return option.is_init_true(self.init_salient_event.get_target_position())
        return False

    def should_complete_chain(self, option):
        """ Check if a newly learned option completes its corresponding chain. """
        assert isinstance(option, Option), f"{type(option)}"

        # If there is a path from a descendant of the chain's init salient event
        # to the newly learned option, then that chain's job is done
        for descendant in self._init_descendants:
            if isinstance(descendant, SalientEvent):
                if self.should_exist_edge_from_event_to_option(descendant, option):
                    self.completing_vertex = descendant, "descendant"
                    return True
            if isinstance(descendant, Option):
                if self.should_exist_edge_between_options(descendant, option):
                    self.completing_vertex = descendant, "descendant"
                    return True

        # If not, then check if there is a direct path from the init-salient-event to the new option
        if self.should_exist_edge_from_event_to_option(self.init_salient_event, option):
            self.completing_vertex = self.init_salient_event, "init_event"
            return True

        # If you intersect with a vertex which is further way from the target than the init-salient-event,
        # then also the chain building can be considered complete
        init_distance = self.init_salient_event.distance_to_other_event(self.target_salient_event)
        for ancestor in self._init_ancestors:
            if isinstance(ancestor, SalientEvent):
                if self.should_exist_edge_from_event_to_option(ancestor, option) and \
                        ancestor.distance_to_other_event(self.target_salient_event) > init_distance:
                    self.completing_vertex = ancestor, "ancestor"
                    return True
            if isinstance(ancestor, Option):
                if self.should_exist_edge_between_options(ancestor, option) and \
                        SalientEvent.set_to_set_distance(option.effect_set, ancestor.effect_set) > init_distance:
                    self.completing_vertex = ancestor, "ancestor"
                    return True

        # Finally, default to the start-state salient event of the entire MDP
        if self.should_exist_edge_from_event_to_option(self.mdp_init_salient_event, option):
            self.completing_vertex = self.mdp_init_salient_event, "mdp_init_salient_event"
            return True

        return False

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
            if self.should_complete_chain(option):
                self._is_deemed_completed = True
                return True

        for option in completed_options:
            if self.should_expand_initiation_classifier(option):
                option.expand_initiation_classifier(self.init_salient_event)
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
        position = state.position if isinstance(state, State) else state[:2]
        assert isinstance(position, np.ndarray), type(position)
        return position

    @staticmethod
    def get_position_matrix(states):
        to_position = lambda s: s.position if isinstance(s, State) else s[:2]
        positions = [to_position(state) for state in states]
        return np.array(positions)
