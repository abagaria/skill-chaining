import pdb
import numpy as np
from simple_rl.agents.func_approx.dsc.OptionClass import Option


class SkillChain(object):
    def __init__(self, start_states, mdp_start_states, target_predicate, options, chain_id, intersecting_options=[]):
        """
        Data structure that keeps track of all options in a particular chain,
        where each chain is identified by a unique target salient event. Chain here
        may also refer to Skill Trees.
        Args:
            start_states (list): List of states at which chaining stops
            mdp_start_states (list): list of MDP start states, if distinct from `start_states`
            target_predicate (function): f: s -> {0, 1} based on salience
            options (list): list of options in the current chain
            chain_id (int): Identifier for the current skill chain
            intersecting_options (list): List of options whose initiation sets overlap
        """
        self.options = options
        self.start_states = start_states
        self.mdp_start_states = mdp_start_states
        self.target_predicate = target_predicate
        self.chain_id = chain_id
        self.intersecting_options = intersecting_options

        if target_predicate is None and len(intersecting_options) > 0:
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

    def _state_in_chain(self, state):
        """ Is state inside the initiation set of any of the options in the chain. """
        for option in self.options:  # type: Option
            if option.initiation_classifier is not None and option.is_init_true(state):
                return True
        return False

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
        # Continue if there is any start state that is not covered
        start_state_in_chain = any([self._state_in_chain(s) for s in self.start_states])

        if self.chain_id == 3:
            return not start_state_in_chain

        assert self.chain_id != 3, self.chain_id

        # Continue if no chain intersections have been found yet
        chain_itersects_another = any([self.is_intersecting(chain) for chain in chains])

        # If chain intersects another, check if at least some other chain has
        # chained all the way back to the start states of the MDP
        mdp_chained = False
        for chain in chains:  # type: SkillChain
            if chain.chain_id != self.chain_id:
                mdp_start_states_in_chain = all([chain._state_in_chain(s) for s in self.mdp_start_states])
                if mdp_start_states_in_chain:
                    mdp_chained = True

        # If the chains do not intersect, continue until the MDP is chained
        stop_condition = start_state_in_chain or (chain_itersects_another and mdp_chained)
        return not stop_condition

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
                if len(my_option.positive_examples) > 0 and len(other_option.positive_examples) > 0:
                    positive_feature_matrix = my_option.construct_feature_matrix(my_option.positive_examples)
                    other_positive_feature_matrix = other_option.construct_feature_matrix(other_option.positive_examples)

                    state_matrix = np.concatenate((positive_feature_matrix, other_positive_feature_matrix), axis=0)
                    my_predictions = my_option.batched_is_init_true(state_matrix)
                    other_predictions = other_option.batched_is_init_true(state_matrix)
                    intersections = np.logical_and(my_predictions, other_predictions)

                    # If at least one state is inside the initiation classifier of both options,
                    # we have found our salient intersection event. Also verify that we have fit initiation
                    # classifiers for both options - this is needed if option's initialize_everywhere property is true
                    if intersections.sum() > 0 and my_option.initiation_classifier is not None and \
                            other_option.initiation_classifier is not None:
                        return my_option, other_option

        return None

    def is_intersecting(self, other_chain):
        """ Boolean wrapper around detect_intersection(). """
        return self.detect_intersection(other_chain) is not None

    def chain_salience_satisfied(self, state):
        return self.target_predicate(state)
