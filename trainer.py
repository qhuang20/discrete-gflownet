"""
Training orchestrator for GFlowNet models.
"""
import os
import datetime
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

try:
    from threadpoolctl import ThreadpoolController
    controller = ThreadpoolController()
    controller.limit(limits=1, user_api='blas')  # eliminates thread contention and CPU thrashing 
except ImportError:
    print("threadpoolctl not available, skipping BLAS thread limitation")

from disc_gflownet.utils.setting import set_seed, set_device
from disc_gflownet.utils.logging import log_arguments, log_training_loop, track_trajectories
from disc_gflownet.agents.tbflownet_agent import TBFlowNetAgent
from disc_gflownet.agents.dbflownet_agent import DBFlowNetAgent
from disc_gflownet.envs.grid_env import GridEnv
from disc_gflownet.envs.grid_env2 import GridEnv2
from disc_gflownet.envs.grid_env_local import GridEnvLocal


def compute_reward(curr_ns, env, reward_func):
    """Compute reward for a single state - original efficient version"""
    curr_ns_state = env.encoding_to_state(curr_ns)
    return reward_func(curr_ns_state) + env.min_reward


class CheckpointManager:
    """Handles saving and loading training checkpoints."""
    
    def __init__(self, run_dir):
        self.run_dir = run_dir
        
    def save(self, agent, optimizer, losses, zs, current_step, 
             ep_last_state_counts, ep_last_state_trajectories, interrupted=False):
        """Save training checkpoint to file"""
        checkpoint = {
            'losses': losses,
            'zs': zs,
            'agent_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'current_step': current_step,
            'ep_last_state_counts': ep_last_state_counts,
            'ep_last_state_trajectories': ep_last_state_trajectories,
        }
        
        filename = 'checkpoint_interrupted.pt' if interrupted else 'checkpoint.pt'
        checkpoint_path = os.path.join(self.run_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        if interrupted:
            print(f"\nTraining interrupted by user.")
            print(f"Checkpoint saved to {checkpoint_path}")
            
    def load(self, filepath):
        """Load checkpoint from file"""
        return torch.load(filepath)


class GFlowNetTrainer:
    """Main trainer class for GFlowNet models."""
    
    def __init__(self, config):
        self.config = config
        self.losses = []
        self.zs = []
        self.agent = None
        self.optimizer = None
        self.envs = None
        self.ep_last_state_counts = {}
        self.ep_last_state_trajectories = {}
        
        # Setup
        self._setup_training()
        self._setup_environments()
        self._setup_agent()
        self._setup_logging()
        
    def _setup_training(self):
        """Initialize training environment"""
        set_seed(self.config.seed)
        set_device(torch.device(self.config.device))
        
    def _setup_environments(self):
        """Create environment instances"""
        if self.config.env_type == 'GridEnv':
            env_class = GridEnv
        elif self.config.env_type == 'GridEnv2':
            env_class = GridEnv2
        elif self.config.env_type == 'GridEnvLocal':
            env_class = GridEnvLocal
        else:
            raise ValueError(f"Unknown environment type: {self.config.env_type}")
        self.envs = [env_class(self.config) for _ in range(self.config.envsize)]
        
    def _setup_agent(self):
        """Initialize agent and optimizer"""
        if self.config.method == 'tb':
            self.agent = TBFlowNetAgent(self.config, self.envs)
            self.optimizer = torch.optim.Adam([
                {'params': self.agent.parameters(), 'lr': self.config.tb_lr}, 
                {'params': [self.agent.log_z], 'lr': self.config.tb_z_lr}
            ])
        elif self.config.method in ['db', 'fldb']:
            self.agent = DBFlowNetAgent(self.config, self.envs)
            self.optimizer = torch.optim.Adam([
                {'params': self.agent.parameters(), 'lr': self.config.tb_lr}
            ])
            
    def _setup_logging(self):
        """Setup logging and checkpointing"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (f"{self.config.method}_h{self.config.n_hid}_l{self.config.n_layers}_"
                   f"mr{self.config.min_reward}_ts{self.config.n_train_steps}_"
                   f"d{self.config.n_dims}_s{self.config.n_steps}_er{self.config.explore_ratio}_"
                   f"{self.config.reward_func_name}")
        
        self.run_dir = os.path.join('runs', f'{timestamp}_{run_name}')
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.checkpoint_manager = CheckpointManager(self.run_dir)
        
        if self.config.log_flag:
            self.log_filename = os.path.join(self.run_dir, 'training.log')
            log_arguments(self.log_filename, self.config)
            
    def _compute_rewards_parallel(self, experiences):
        """Compute rewards using multiprocessing - restored original efficient version"""
        if self.config.n_workers <= 1:
            return
            
        # Use numpy arrays like original code for better compatibility
        curr_ns_all = np.zeros((self.config.mbsize, self.config.n_steps, self.envs[0].encoding_dim))
        for mb in range(self.config.mbsize):
            curr_ns_all[mb] = experiences[0][mb].cpu()[1:]
        curr_ns_all = curr_ns_all.reshape(self.config.mbsize * self.config.n_steps, self.envs[0].encoding_dim)
        
        # Use partial function with existing environment (original efficient approach)
        compute_reward_partial = partial(compute_reward, env=self.envs[0], reward_func=self.config.custom_reward_fn)
        
        with Pool(processes=self.config.n_workers) as env_pool:
            curr_r_env = list(env_pool.imap(compute_reward_partial, curr_ns_all))
        
        # Convert to numpy array then reshape, like original
        curr_r_env = np.asarray(curr_r_env)
        curr_r_env = curr_r_env.reshape(self.config.mbsize, self.config.n_steps, 1)
        
        # Convert back to torch tensors for experiences
        for mb in range(self.config.mbsize):
            experiences[3][mb] = torch.from_numpy(curr_r_env[mb])
            
    def train_step(self, step):
        """Execute a single training step"""
        experiences = self.agent.sample_batch_episodes(self.config.mbsize)
        self._compute_rewards_parallel(experiences)
        
        # Compute loss
        if self.config.method == 'fldb':
            loss, z = self.agent.compute_batch_loss(experiences, use_fldb=True)
        else:
            loss, z = self.agent.compute_batch_loss(experiences)
            
        self.losses.append(loss.item())
        self.zs.append(z.item())
        
        # Optimization step
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Tracking and logging
        track_trajectories(experiences, self.envs[0], self.ep_last_state_counts, 
                         self.ep_last_state_trajectories, step)
        
        if step % self.config.log_freq == 0:
            if self.config.log_flag:
                log_training_loop(self.log_filename, self.agent, step, 
                                self.ep_last_state_counts, self.ep_last_state_trajectories)
            self.checkpoint_manager.save(self.agent, self.optimizer, self.losses, self.zs, 
                                       step, self.ep_last_state_counts, self.ep_last_state_trajectories)
            
    def train(self):
        """Main training loop"""
        try:
            for step in tqdm(range(self.config.n_train_steps + 1), disable=not self.config.progress):
                self.train_step(step)
                
        except KeyboardInterrupt:
            self.checkpoint_manager.save(self.agent, self.optimizer, self.losses, self.zs, 
                                       step, self.ep_last_state_counts, self.ep_last_state_trajectories, 
                                       interrupted=True)
            print("Training interrupted by user")
            return
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            raise
            
        return {
            'losses': self.losses,
            'zs': self.zs,
            'agent': self.agent,
            'ep_last_state_counts': self.ep_last_state_counts,
            'ep_last_state_trajectories': self.ep_last_state_trajectories
        } 