import pdb
import numpy as np
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent


class SkillChain(object):
    def __init__(self, start_states, mdp_start_states, init_salient_event, target_salient_event, options, chain_id,
                 intersecting_options=[], is_backward_chain=False, has_backward_chain=False,
                 option_intersection_salience=False, event_intersection_salience=True):
        """
        Data structure that keeps track of all options in a particular chain,
        where each chain is identified by a unique target salient event. Chain here
        may also refer to Skill Trees.
        Args:
            start_states (list): List of states at which chaining stops
            mdp_start_states (list): list of MDP start states, if distinct from `start_states`
            init_salient_event (SalientEvent): f: s -> {0, 1} based on start salience
            target_salient_event (SalientEvent): f: s -> {0, 1} based on target salience
            options (list): list of options in the current chain
            chain_id (int): Identifier for the current skill chain
            intersecting_options (list): List of options whose initiation sets overlap
            is_backward_chain (bool): Whether this chain goes from start -> salient or from salient -> start
            has_backward_chain (bool): Does there exist a backward chain corresponding to the current forward chain (N/A if `is_backward_chain`)
            option_intersection_salience (bool): Whether to chain until the current chain intersects with another option
            event_intersection_salience (bool): Chain until you intersect with another salient event
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
        self.option_intersection_salience = option_intersection_salience
        self.event_intersection_salience = event_intersection_salience

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

        required_to_chain_until_intersection = self.option_intersection_salience or self.event_intersection_salience

        if self.is_backward_chain or not required_to_chain_until_intersection:
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
    def detect_intersection_between_options(my_option, other_option):
        if my_option.get_training_phase() == "initiation_done" and other_option.get_training_phase() == "initiation_done":
            effect_set = other_option.effect_set  # list of states
            effect_set_matrix = SkillChain.get_position_matrix(effect_set)
            inits = my_option.batched_is_init_true(effect_set_matrix)
            is_intersecting = inits.all()
            return is_intersecting
        return False

    @staticmethod
    def detect_intersection_between_option_and_event(option, event):
        if option.get_training_phase() == "initiation_done":
            if len(event.trigger_points) > 0:  # Be careful: all([]) = True
                state_matrix = SkillChain.get_position_matrix(event.trigger_points)
                inits = option.batched_is_init_true(state_matrix)
                is_intersecting = inits.all()
                return is_intersecting
            return option.is_init_true(event.target_state)
        return False

    def detect_intersection_with_other_chains(self, other_chains):
        for chain in other_chains:
            intersecting_options = self.get_intersecting_options(chain)
            if intersecting_options is not None:
                return intersecting_options
        return None

    def get_intersecting_options(self, other_chain):
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

    def get_intersecting_option_and_event(self, other_chain):
        # Do not identify self-intersections
        if self == other_chain:
            return None

        for my_option in self.options:  # type: Option
            event = other_chain.target_salient_event
            if self.detect_intersection_between_option_and_event(my_option, event):
                return my_option, event
        return None

    def is_intersecting(self, other_chain):
        """ Boolean wrapper around detect_intersection(). """
        if self.option_intersection_salience:
            return self.get_intersecting_options(other_chain) is not None
        assert self.event_intersection_salience, "set event_intersection_salience or option_intersection_salience"
        return self.get_intersecting_option_and_event(other_chain) is not None

    def chain_salience_satisfied(self, state):
        if self.target_salient_event is not None:
            return self.target_salient_event(state)
        assert len(self.intersecting_options) > 0, self.intersecting_options
        return self.target_predicate(state)

    def is_chain_completed(self, chains):
        """
        Either we are chained till the start state or we are chained till another event
        which already has some chaining targeting it. The reason we only check for
        intersection with chained events is that we want to before we re-wire the current
        chain, we want to be sure that we have a way to trigger its salient event.
        Args:
            chains (list)

        Returns:
            is_completed (bool)
        """
        is_intersecting_another_chain = False
        if self.option_intersection_salience or self.event_intersection_salience:
            other_chains = [chain for chain in chains if chain != self]
            is_intersecting_another_chain = any([self.is_intersecting(chain) and chain.is_chain_completed(other_chains)
                                                 for chain in other_chains])
        return self.chained_till_start_state() or is_intersecting_another_chain

    def chained_till_start_state(self):
        start_state_in_chain = all([self.state_in_chain(s) for s in self.start_states])
        return start_state_in_chain

    def get_overlapping_options_and_events(self, chains):
        """ Options in the current chain that satisfy the target salient events of other chains. """
        def is_linked(e, o):
            """ Given a target salient event from another chain and an option,
                tell me if the event is a subset of the option's initiation set. """
            if o.get_training_phase() == "initiation_done":
                if len(e.trigger_points) > 0: # Be careful: all([]) = True
                    return all([o.is_init_true(s) for s in e.trigger_points])
                return o.is_init_true(e.target_state)
            return False

        links = []
        events = [chain.target_salient_event for chain in chains if chain != self]

        for event in events:
            for option in self.options:
                if is_linked(event, option):
                    links.append((event, option))

        return links

    def get_leaf_nodes_from_skill_chain(self):
        return [option for option in self.options if len(option.children) == 0]

    def get_root_nodes_from_skill_chain(self):
        return [option for option in self.options if option.parent is None]

    @staticmethod
    def get_position_matrix(states):
        positions = [state.position for state in states]
        return np.array(positions)
