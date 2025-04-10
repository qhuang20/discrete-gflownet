import numpy as np


def track_trajectories(batch_data, env, ep_last_state_counts, ep_last_state_trajectories, training_step):
    """Track unique state distribution and trajectory rewards"""
    batch_ss, batch_as, batch_steps, batch_rs = batch_data
    
    for i in range(len(batch_rs)):
        # Get trajectory data
        ep_states_encoded = batch_ss[i].cpu().data.numpy()
        ep_actions = batch_as[i].cpu().data.numpy()
        ep_rewards = batch_rs[i].cpu().data.numpy()
        
        # Print shapes
        # print(f"ep_states_encoded shape: {ep_states_encoded.shape}")
        # print(f"ep_actions shape: {ep_actions.shape}")
        # print(f"ep_rewards shape: {ep_rewards.shape}")
        # exit()
        
        # Convert encoded states to actual states
        ep_states = []
        for encoded_state in ep_states_encoded:
            state = env.encoding_to_state(encoded_state)
            if env.enable_time:
                state = tuple(state[1])  # Only keep spatial coordinates
            else:
                state = tuple(state)
            ep_states.append(state)
        
        # Get final state
        env_state = ep_states[-1]
            
        # Update counts
        if env_state in ep_last_state_counts:
            ep_last_state_counts[env_state] += 1
        else:
            ep_last_state_counts[env_state] = 1
            ep_last_state_trajectories[env_state] = []
            
        # Add initial state reward of 0 to match state length
        padded_rewards = np.vstack(([0], ep_rewards))
            
        # Store trajectory
        trajectory = {
            'states': ep_states,
            'actions': ep_actions,
            'rewards': padded_rewards,
            'training_step': training_step
        }
        ep_last_state_trajectories[env_state].append(trajectory)
        

def log_arguments(log_filename, args):
    with open(log_filename, 'w') as f:
        f.write("Training Arguments:\n")
        f.write("-" * 50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("-" * 50 + "\n\n")
        
def log_training_loop(log_filename, agent, step, ep_last_state_counts, ep_last_state_trajectories):
    with open(log_filename, 'a') as f:
        print("\n", file=f)
        print("-" * 100, file=f)
        print("-" * 100, file=f)
        print("-" * 100, file=f)
        print(f"Step: {step}", file=f)
        print(f"Number of unique terminal states found: {len(ep_last_state_counts)}", file=f)
        
        print("-" * 30, file=f)
        print("Top 10 by Avg avg trajectory rewards:", file=f)
        print("-" * 30, file=f)
        
        # Calculate the average of average trajectory rewards for each state 
        state_avg_rewards = {}
        for state, trajectories in ep_last_state_trajectories.items():
            traj_avgs = []
            for traj in trajectories:
                rewards = [r[0] for r in traj['rewards']]
                traj_avgs.append(sum(rewards) / len(rewards))
            state_avg_rewards[state] = sum(traj_avgs) / len(traj_avgs)
        
        # Sort states by state_avg_rewards 
        top_reward_states = sorted(
            state_avg_rewards.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for state, avg_reward in top_reward_states:
            trajectories = ep_last_state_trajectories[state]
            count = ep_last_state_counts[state]
            terminal_reward = trajectories[0]['rewards'][-1][0]
            print(f"State: {state}, Count: {count}, Terminal reward: {terminal_reward:.3f}, Avg avg trajectory reward: {avg_reward:.3f}", file=f)
            
            # Print each trajectory and its average reward
            for traj in trajectories:
                rewards = [r[0] for r in traj['rewards']]
                states = traj['states']
                traj_avg = sum(rewards) / len(rewards)
                state_reward_pairs = []
                for i, (state, reward) in enumerate(zip(states, rewards)):
                    if reward > 1:
                        state_reward_pairs.append(f"({state}, ${reward:.3f}$)")
                    else:
                        state_reward_pairs.append(f"({state}, {reward:.3f})")
                print(f"Trajectory state-reward pairs: {state_reward_pairs}, Average: {traj_avg:.3f}", file=f)
            print("", file=f)
        print("\n", file=f)
        
        
        # print("-" * 30, file=f)
        # print("Top 10 by visit count:", file=f)
        # print("-" * 30, file=f)
        
        # top_count_states = sorted(
        #     ep_last_state_counts.items(),
        #     key=lambda x: x[1],
        #     reverse=True
        # )[:10]
        
        # for state, count in top_count_states:
        #     trajectories = ep_last_state_trajectories[state]
        #     terminal_reward = trajectories[0]['rewards'][-1][0]
        #     print(f"State: {state}, Count: {count}, Terminal reward: {terminal_reward:.3f}, Avg avg trajectory reward: {state_avg_rewards[state]:.3f}", file=f)
            
        #     # Print each trajectory and its average reward
        #     for traj in trajectories:
        #         rewards = [r[0] for r in traj['rewards']]
        #         traj_avg = sum(rewards) / len(rewards)
        #         print(f"Trajectory rewards: {[f'{r:.3f}' for r in rewards]}, Average: {traj_avg:.3f}", file=f)
        #     print("", file=f)
        # print("\n", file=f)


