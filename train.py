import numpy as np
import networkx as nx
import pickle



def main(args):
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    import json
    
    parser = ArgumentParser(description='GFlowNet for Genetic Circuits Design.')
    args = parser.parse_args()

    main(args)
    
    
    


def main(args):
    global all_losses  
    global logZs
    global agent
    
    set_seed(args.seed)
    args.dev = torch.device(args.device)
    set_device(args.dev)
    method_name = args.method

    # # explore_ratio is fine-tuned for each method by grid search
    # if method_name == 'tb_gfn':
    #     args.explore_ratio = 0.0625
    # elif method_name == 'db_gfn':
    #     args.explore_ratio = 0.0625 # 0.1
    # elif method_name == 'fldb_gfn':
    #     args.explore_ratio = 0.0625 # 0.5



    
    envs = [SetEnv(args) for i in range(args.envsize)]

    if args.method == 'tb_gfn':
        agent = TBFlowNetAgent(args, envs)
        opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}, {'params':[agent.Z], 'lr': args.tb_z_lr} ])
    elif args.method == 'db_gfn' or args.method == 'fldb_gfn':
        agent = DBFlowNetAgent(args, envs)
        opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}])




    
    
    
    """train"""
    
    all_losses = [] 
    logZs = []  # only for tb_gfn
    for i in tqdm(range(args.n_train_steps + 1), disable=not args.progress):
        # print("\n" + "-"*30 + "training step: " + str(i) + "-"*30)
        experiences = agent.sample_batch_episodes(args.mbsize)
        # print(len(experiences))
        # print(experiences)
        if method_name == 'fldb_gfn':
            losses = agent.compute_batch_loss(experiences, use_fldb=True) 
        else:
            losses = agent.compute_batch_loss(experiences) 
            
        all_losses.append(losses[0].item())
        logZs.append(losses[1].item()) 

        losses[0].backward()
        opt.step()
        opt.zero_grad()
        
        if i % args.log_freq == 0:
            print("--------------------------------")
            print("\nStep", i)
            print("Number of unique states found:", len(agent.ep_last_state_counts)) 
            print("\nTop 12 states by reward:")
            top_states = sorted(agent.ep_last_state_ep_rewards.items(), 
                              key=lambda x: x[1][-1], # Sort by last reward in trajectory
                              reverse=True)[:12]
            for state in top_states:
                ep_rewards = [f"{r[0]:.3f}" for r in agent.ep_last_state_ep_rewards[state[0]]]
                count = agent.ep_last_state_counts[state[0]]
                print(f"Full trajectory rewards: {ep_rewards}, Count: {count}, State: {state[0]}")
            print("\n")





# if args.size == 'tiny':
#     args.action_dim = 3
#     args.set_size = 2
#     intermediate_energies = interm_energies['tiny']
# elif args.size == 'small':
#     args.action_dim = 30
#     args.set_size = 20
#     intermediate_energies = interm_energies['small']
# elif args.size == 'medium':
#     args.action_dim = 80
#     args.set_size = 60
#     intermediate_energies = interm_energies['medium']
# elif args.size == 'large':
#     args.action_dim = 100
#     args.set_size = 80
#     intermediate_energies = interm_energies['large']



# Set up argparse arguments manually
args = argparse.Namespace(
    device='cpu',
    progress=True,
    seed=0,
    n_train_steps=1000,  # 2000
    log_freq=1000,
    mbsize=16, #16
    # Model
    method='db_gfn',
    learning_rate=1e-4,
    tb_lr=0.001,
    tb_z_lr=0.1,
    n_hid=256,
    n_layers=2,
    explore_ratio=0.2,
    temp=1.,
    uni_rand_pb=1,
    # Env
    envsize=16, #16
    custom_reward_fn=set_custom_reward_func,
    reward_set_size='tiny',
    action_dim=3,
    set_size=2,
    
)

# Call the main function
main(args)
