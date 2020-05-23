import pdb
import numpy as np
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.BaseSalientEventClass import BaseSalientEvent


class SkillChain(object):
    def __init__(self, start_states, mdp_start_states, init_salient_event, target_salient_event, options, chain_id,
                 intersecting_options=[], is_backward_chain=False, has_backward_chain=False, chain_until_intersection=False):
        """
        Data structure that keeps track of all options in a particular chain,
        where each chain is identified by a unique target salient event. Chain here
        may also refer to Skill Trees.
        Args:
            start_states (list): List of states at which chaining stops
            mdp_start_states (list): list of MDP start states, if distinct from `start_states`
            init_salient_event (BaseSalientEvent): f: s -> {0, 1} based on start salience
            target_salient_event (BaseSalientEvent): f: s -> {0, 1} based on target salience
            options (list): list of options in the current chain
            chain_id (int): Identifier for the current skill chain
            intersecting_options (list): List of options whose initiation sets overlap
            is_backward_chain (bool): Whether this chain goes from start -> salient or from salient -> start
            has_backward_chain (bool): Does there exist a backward chain corresponding to the current forward chain (N/A if `is_backward_chain`)
            chain_until_intersection (bool): Whether to chain until the current chain intersects with another OR simply until the start states
        """
        self.options = options
        self.start_states = start_states
        self.mdp_start_states = mdp_start_states
        self.init_salient_event = init_salient_event  # TODO: USE THIS IN PLACE OF START STATES
        self.target_salient_event = target_salient_event
        self.target_position = target_salient_event.target_state
        self.chain_id = chain_id
        self.intersecting_options = intersecting_options

        self.is_backward_chain = is_backward_chain
        self.has_backward_chain = has_backward_chain
        self.chain_until_intersection = chain_until_intersection

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

    def state_in_chain(self, state):
        """ Is state inside the initiation set of any of the options in the chain. """
        for option in self.options:  # type: Option
            if option.initiation_classifier is not None and \
                    option.is_init_true(state) and option.get_training_phase() == "initiation_done":
                return True
        return False

    def get_option_for_state(self, state):
        for option in self.options:  # type: Option
            if option.initiation_classifier is not None and \
                    option.is_init_true(state):
                return option
        return None

    def should_continue_chaining(self, chains):
        """
        Determine if we should keep learning skills to add to the current chain.
        We keep discovering skills in the current chain as long as one any start state
        is not inside the initiation set of an option OR if we have chained back to
        another skill chain - in which case we will also have an intersection salient event.
        Args:
            chains (list): List of SkillChain objects so we can check for intersections

        Returns:
            should_create (bool): return True if there is some start_state that is
                                  not inside any of the options in the current chain
        """
        # Continue if not all the start states have been covered by the options in the current chain
        start_state_in_chain = self.chained_till_start_state()

        if self.is_backward_chain or not self.chain_until_intersection:
            return not start_state_in_chain

        # For forward chains, continue until chain intersections have been found yet
        chain_itersects_another = any([self.is_intersecting(chain) for chain in chains])

        # If chain intersects another, check if at least some other chain has
        # chained all the way back to the start states of the MDP
        mdp_chained = False
        for chain in chains:  # type: SkillChain
            if chain.chain_id != self.chain_id:
                mdp_start_states_in_chain = all([chain.state_in_chain(s) for s in self.mdp_start_states])
                if mdp_start_states_in_chain:
                    mdp_chained = True

        # If the chains do not intersect, continue until the MDP is chained
        stop_condition = start_state_in_chain or (chain_itersects_another and mdp_chained)
        return not stop_condition

    @staticmethod
    def get_positive_states_from_options(option1, option2):
        positive_feature_matrix = option1.construct_feature_matrix(option1.positive_examples)
        other_positive_feature_matrix = option2.construct_feature_matrix(option2.positive_examples)

        state_matrix = np.concatenate((positive_feature_matrix, other_positive_feature_matrix), axis=0)
        return state_matrix

    @staticmethod
    def get_intersecting_states_between_options(my_option, other_option):
        state_matrix = SkillChain.get_positive_states_from_options(my_option, other_option)
        intersections = SkillChain.get_intersecting_indices(my_option, other_option)
        if intersections.sum() > 0 and my_option.get_training_phase() == "initiation_done" and \
            other_option.get_training_phase() == "initiation_done":
            return state_matrix[intersections == 1, :]
        return []

    @staticmethod
    def get_intersecting_indices(my_option, other_option):
        state_matrix = SkillChain.get_positive_states_from_options(my_option, other_option)
        my_predictions = my_option.batched_is_init_true(state_matrix)
        other_predictions = other_option.batched_is_init_true(state_matrix)
        intersections = np.logical_and(my_predictions, other_predictions)
        return intersections

    @staticmethod
    def detect_intersection_between_options(my_option, other_option):
        if len(my_option.positive_examples) > 0 and len(other_option.positive_examples) > 0:

            intersections = SkillChain.get_intersecting_indices(my_option, other_option)

            # If at least one state is inside the initiation classifier of both options,
            # we have found our salient intersection event. Also verify that we have fit initiation
            # classifiers for both options - this is needed if option's initialize_everywhere property is true
            if intersections.sum() > 0 and my_option.get_training_phase() == "initiation_done" and \
                    other_option.get_training_phase() == "initiation_done":
                return True
        return False

    def detect_intersection_with_other_chains(self, other_chains):
        for chain in other_chains:
            intersecting_options = self.detect_intersection(chain)
            if intersecting_options is not None:
                return intersecting_options
        return None

    def detect_intersection(self, other_chain):
        """
        One chain intersecting with another defines a chain intersection event.
        The intersecting region is treated as a salient event to construct skill graphs.
        To detect an intersection between two chains, we iterate through all the options
        in the two chains and look for states that belong in both chains.
        Args:
            other_chain (SkillChain)

        Returns:
            option_pair (tuple): pair of intersecting options if intersection event is
                                 identified, else return None
        """

        # Do not identify self-intersections
        if self == other_chain:
            return None

        for my_option in self.options:  # type: Option
            for other_option in other_chain.options:  # type: Option
                if self.detect_intersection_between_options(my_option, other_option):
                        return my_option, other_option
        return None

    def is_intersecting(self, other_chain):
        """ Boolean wrapper around detect_intersection(). """
        return self.detect_intersection(other_chain) is not None

    def chain_salience_satisfied(self, state):
        if self.target_salient_event is not None:
            return self.target_salient_event(state)
        assert len(self.intersecting_options) > 0, self.intersecting_options
        return self.target_predicate(state)

    def chained_till_start_state(self):
        start_state_in_chain = all([self.state_in_chain(s) for s in self.start_states])
        return start_state_in_chain

    def get_leaf_nodes_from_skill_chain(self):
        return [option for option in self.options if len(option.children) == 0]

    def get_root_nodes_from_skill_chain(self):
        return [option for option in self.options if option.parent is None]
