import sys
import os
import subprocess
import datetime
import signal
from multiprocessing import Pool

def run_experiment(n_dims):
    """Run a single experiment with specified n_dims"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"runs/{timestamp}_ndims{n_dims}_output.log"
    
    cmd = [
        "python", 
        "train.py",
        "--mbsize", "8",
        "--envsize", "8", 
        "--method", "tb",
        "--n_workers", "1",
        "--n_hid", "256",
        "--n_layers", "3",
        "--explore_ratio", "0.35",
        "--n_steps", "15",  
        "--n_train_steps", "1000",
        "--n_dims", str(n_dims),
    ]
    
    print(f"\nStarting experiment with n_dims={n_dims}")
    
    # Ensure runs directory exists
    os.makedirs("runs", exist_ok=True)
    
    # Redirect output to log file
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True  # Create new session for process group
        )
        try:
            process.wait()
        except:
            # Kill entire process group if interrupted
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            raise
    
    if process.returncode != 0:
        print(f"Error in experiment n_dims={n_dims}. Check {log_file} for details.")
    else:
        print(f"Completed experiment with n_dims={n_dims}. Output saved to {log_file}")

def main():
    # Define experiments
    n_dims_list = [9, 25, 49, 81, 225, 441]
    
    # Number of parallel processes (adjust based on CPU cores)
    n_processes = min(len(n_dims_list), os.cpu_count())
    
    print(f"Running {len(n_dims_list)} experiments using {n_processes} processes")
    
    try:
        # Run experiments in parallel, like nohup
        with Pool(processes=n_processes) as pool:
            pool.map(run_experiment, n_dims_list)
    except KeyboardInterrupt:
        print("\nTerminating all experiments...")
        # The processes will be terminated by the exception handling in run_experiment

if __name__ == "__main__":
    main()



