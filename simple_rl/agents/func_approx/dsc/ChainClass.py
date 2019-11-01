import numpy as np
from simple_rl.agents.func_approx.dsc.OptionClass import Option


class SkillChain(object):
    def __init__(self, target_predicate, options, chain_id):
        """
        Data structure that keeps track of all options in a particular chain,
        where each chain is identified by a unique target salient event. Chain here
        may also refer to Skill Trees.
        Args:
            target_predicate (function): f: s -> {0, 1} based on salience
            options (list): list of options in the current chain
            chain_id (int): Identifier for the current skill chain
        """
        self.options = options
        self.target_predicate = target_predicate
        self.chain_id = chain_id

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
            if option.is_init_true(state):
                return True
        return False

    def should_continue_chaining(self, start_states, other_chains):
        """
        Determine if we should keep learning skills to add to the current chain.
        We keep discovering skills in the current chain as long as one any start state
        is not inside the initiation set of an option OR if we have chained back to
        another skill chain - in which case we will also have an intersection salient event.
        Args:
            start_states (list): List of states to which we want to chain the current set of options
            other_chains (list): List of SkillChain objects so we can check for intersections

        Returns:
            should_create (bool): return True if there is some start_state that is
                                  not inside any of the options in the current chain
        """
        # Continue if there is any start state that is not covered
        start_state_in_chain = any([not self._state_in_chain(s) for s in start_states])

        # Continue if no chain intersections have been found yet
        chain_itersects_another = any([self.is_intersecting(chain) for chain in other_chains])

        # Stop chaining if either condition says to stop
        return not start_state_in_chain and not chain_itersects_another

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
        for my_option in self.options:  # type: Option
            for other_option in other_chain.options:  # type: Option
                positive_feature_matrix = my_option.construct_feature_matrix(my_option.positive_examples)
                other_positive_feature_matrix = other_option.construct_feature_matrix(other_option.positive_examples)

                state_matrix = np.concatenate((positive_feature_matrix, other_positive_feature_matrix), axis=0)
                my_predictions = my_option.batched_is_init_true(state_matrix)
                other_predictions = other_option.batched_is_init_true(state_matrix)
                intersections = np.logical_and(my_predictions, other_predictions)

                # If at least one state is inside the initiation classifier of both options,
                # we have found our salient intersection event.
                if intersections.sum() > 0:
                    return my_option, other_option

        return None

    def is_intersecting(self, other_chain):
        """ Boolean wrapper around detect_intersection(). """
        return self.detect_intersection(other_chain) is not None
