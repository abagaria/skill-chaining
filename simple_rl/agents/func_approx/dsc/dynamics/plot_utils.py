import torch
import numpy as np
from copy import deepcopy

def model_trajectory_prediction_error(model, mdp, num_rollouts, horizon):
    model_traj2D = []
    act_traj2D = []
    with torch.no_grad():
        for _ in range(num_rollouts):
            mdp.reset()
            s = mdp.cur_state
            model_traj = []
            act_traj = []
            action_history = []
            state = torch.from_numpy(np.array([s])).cuda()
            for i in range(horizon):
                a = mdp.sample_random_action() 
                action_history.append(a)
                action = torch.from_numpy(np.array([a])).cuda()
                next_state = model.predict_next_state(state.float(), action.float())
                
                state = next_state
                model_traj.append(state.cpu().squeeze().numpy()[:2])
            
            for action in action_history:
                mdp.execute_agent_action(action)
                s = deepcopy(mdp.cur_state)
                act_traj.append(s.position)
            
            model_traj2D.append(model_traj)
            act_traj2D.append(act_traj)
    return np.array(model_traj2D), np.array(act_traj2D)

def num_steps_to_reach_goal(model, mdp, goal_x, goal_y, thres, num_rollouts, num_steps, runs):
    tracker = []
    with torch.no_grad():
        results = np.zeros((num_rollouts, num_steps))
        for _ in range(runs):
            mdp.reset()
            s = mdp.cur_state
            s = mdp.cur_state
            sx, sy = s.position
            count = 0
            while abs(sx - goal_x) >= thres or abs(sy - goal_y) >= thres:
                # sample actions for all steps
                np_actions = np.zeros((num_rollouts, num_steps, 8))
                np_states = np.repeat(np.array([s]), num_rollouts, axis=0)
                for i in range(num_rollouts):
                    for j in range(num_steps):
                        np_actions[i,j,:] = mdp.sample_random_action()
                
                # compute next states for each step
                for j in range(num_steps):
                    actions = np_actions[:,j,:]
                    states_t = torch.from_numpy(np_states)
                    actions_t= torch.from_numpy(actions)
                    # transfer to gpu
                    states_t = states_t.cuda()
                    actions_t = actions_t.cuda()

                    pred = model.predict_next_state(states_t.float(), actions_t.float())
                    np_states = pred.cpu().numpy()
                    # update results with whatever distance metric
                    results[:,j] = (goal_x - np_states[:,0]) ** 2 + (goal_y - np_states[:,1]) ** 2
                
                # choose next action to execute
                gammas = np.power(0.95 * np.ones(num_steps), np.arange(0, num_steps))
                summed_results = np.sum(results * gammas, axis=1)
                index = np.argmin(summed_results) # retrieve action with least trajectory distance to goal
                action = np_actions[index,0,:] # grab action corresponding to least distance
                
                # execute action in mdp
                mdp.execute_agent_action(action)
                count += 1
                
                # update s for new state
                s = mdp.cur_state
                sx, sy = s.position
                                
                print("action taken!")
            tracker.append(count)
            print("reached goal!")
    return tracker

def num_steps_per_horizon(model, mdp, goal_x, goal_y, thres, num_rollouts, min_horizon, max_horizon, runs_per=5):
    num_steps_per_horizon = []
    for num_steps in range(min_horizon, max_horizon + 1):
        tracker = num_steps_to_reach_goal(model, mdp, goal_x, goal_y, thres, num_rollouts, num_steps, runs_per)
        num_steps_per_horizon.append(tracker)
        print("horizon {} done!".format(num_steps))
    return num_steps_per_horizon