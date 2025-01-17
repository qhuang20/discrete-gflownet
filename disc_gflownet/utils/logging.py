

def log_arguments(log_filename, args):
    with open(log_filename, 'w') as f:
        f.write("Training Arguments:\n")
        f.write("-" * 50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("-" * 50 + "\n\n")
        
def log_training_loop(log_filename, agent, step):
    with open(log_filename, 'a') as f:
        print("\n", file=f)
        print("-" * 60, file=f)
        print(f"Step: {step}", file=f)
        print(f"Number of unique terminal states found: {len(agent.ep_last_state_counts)}", file=f)
        
        print("-" * 30, file=f)
        print("Top 10 by Avg avg trajectory rewards:", file=f)
        print("-" * 30, file=f)
        
        # Calculate the average of average trajectory rewards for each state 
        state_avg_rewards = {}
        for state, trajectories in agent.ep_last_state_trajectories.items():
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
            trajectories = agent.ep_last_state_trajectories[state]
            count = agent.ep_last_state_counts[state]
            terminal_reward = trajectories[0]['rewards'][-1][0]
            print(f"State: {state}, Count: {count}, Terminal reward: {terminal_reward:.3f}, Avg avg trajectory reward: {avg_reward:.3f}", file=f)
            
            # Print each trajectory and its average reward
            for traj in trajectories:
                rewards = [r[0] for r in traj['rewards']]
                traj_avg = sum(rewards) / len(rewards)
                print(f"Trajectory rewards: {[f'{r:.3f}' for r in rewards]}, Average: {traj_avg:.3f}", file=f)
            print("", file=f)
            
        print("-" * 30, file=f)
        print("Top 10 by visit count:", file=f)
        print("-" * 30, file=f)
        
        top_count_states = sorted(
            agent.ep_last_state_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for state, count in top_count_states:
            trajectories = agent.ep_last_state_trajectories[state]
            terminal_reward = trajectories[0]['rewards'][-1][0]
            print(f"State: {state}, Count: {count}, Terminal reward: {terminal_reward:.3f}, Avg avg trajectory reward: {state_avg_rewards[state]:.3f}", file=f)
            
            # Print each trajectory and its average reward
            for traj in trajectories:
                rewards = [r[0] for r in traj['rewards']]
                traj_avg = sum(rewards) / len(rewards)
                print(f"Trajectory rewards: {[f'{r:.3f}' for r in rewards]}, Average: {traj_avg:.3f}", file=f)
            print("", file=f)
        print("\n", file=f)


