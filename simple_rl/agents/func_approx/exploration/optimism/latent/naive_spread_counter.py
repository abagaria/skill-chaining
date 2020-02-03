"""
This doesn't project anything, it just calculates pairwise distance over
EVERY visited state. Then it uses that as a replay buffer thing.

We'll need a function that can score a series of states given a buffer of
"seen" things.


Difficulties: It's probably pretty easy to make something unbiased in "n". How do you
make it unbiased in "n^2" or (1/sqrt(n))?

Saket made a pretty good point that you may be able to figure
the cost out by making one normal (the point you care about) and getting distances
to the other guys. Probably an equivalent way of thinking about it. Actually,
definitely. Akhil said the same. Smart guys.

"""

import numpy as np


# def get_distance_to_buffer(state, buffer):
#     buffer_distance = buffer - state.reshape(1,-1)
#     buffer_distance = (buffer_distance ** 2).sum(axis=1)
#     return buffer_distance[0]

def torch_get_square_distances_to_buffer(states, buffer):
    """

    Args:
        states (torch.tensor): the phi-ed states
        buffer (torch.tensor): a phi-ed action buffer

    Returns:
        a tensor of size (num_states x num_buffer) that represents pairwise distances.

    """
    num_states = states.shape[0]
    num_buffers = buffer.shape[0]

    new_states = states.view(num_states, 1, -1)
    new_buffers = buffer.view(1 ,num_buffers, -1)

    new_states = new_states.expand(-1, num_buffers, -1)
    new_buffers = new_buffers.expand(num_states, -1, -1)

    difference = new_buffers - new_states
    distances = (difference ** 2).sum(dim=-1)

    assert distances.shape == (num_states, num_buffers), distances.shape

    return distances


def get_all_distances_to_buffer(states, buffer):
    """
    In some sense, this should make a a x b x n array,
    where a is num states, b is num buffer states, and n
    is state dim. Then you do mean distance to that.

    One way to get this to work is to make each array the right
    shape and then subtract them. That's what I'll do.

    """
    assert isinstance(states, np.ndarray)
    assert isinstance(buffer, np.ndarray)

    assert states.shape[1] == buffer.shape[1]
    assert len(states.shape) == len(buffer.shape) == 2

    num_features = states.shape[1]
    num_states = states.shape[0]
    num_buffers = buffer.shape[0]

    new_states = states.reshape(num_states, 1, num_features)
    new_states = np.repeat(new_states, num_buffers, axis=1)

    new_buffers = buffer.reshape(1, num_buffers, num_features)
    new_buffers = np.repeat(new_buffers, num_states, axis=0)

    difference = new_buffers - new_states

    # import ipdb; ipdb.set_trace()
    # print(f"{difference.shape}")
    assert len(difference.shape) == 3, len(difference.shape)

    distances = (difference ** 2).sum(axis=-1)

    distances = np.sqrt(distances)

    assert len(distances.shape) == 2, len(distances.shape)
    return distances




def torch_get_all_distances_to_buffer(states, buffer):
    
    pass


def score_state(state, buffer):
    pass

def score_many_states(states, buffer):
    pass



if __name__ == '__main__':
    print("Heelo")
    # import ipdb; ipdb.set_trace()
    s1 = np.asarray([[1,2,3,4,5]])
    s2 = np.asarray([[1,2,3,4,5]])
    asdf = get_all_distances_to_buffer(s1,s2)
    print(asdf)
    print('hingerster')