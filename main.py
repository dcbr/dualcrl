import argparse
import multiprocessing as mp
import shutil
import time

from job import Job


def load_torch():
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Numpy import errors on some architectures without this
    import torch
    torch.set_num_threads(1)  # Multiprocessing goes crazy slow without this


# Argument defaults and allowed values:
procs = 4
modes = ["train", "analyze"]


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning Duality Experiments")
    parser.add_argument('--procs', default=procs, type=int, metavar='P', help="Number of processes to use for the experiments (default: %(default)d)")
    parser.add_argument('--job', type=str, dest='job_name', metavar='NAME', help="Name of the job file to run")
    parser.add_argument('--uid', default=None, type=str, metavar='UID', help="Unique identifier of this job run (default: current timestamp)")
    parser.add_argument('--mode', default="train", type=str, choices=modes, metavar='MODE', help="Run mode: train or analyze (default: %(default)s)")
    args = parser.parse_args()
    if args.uid is None:
        args.uid = int(time.time())
    return args


def execute_run(job_name, job_uid, run_name, mode):
    # This function is executed in a separate process if multiprocessing is enabled
    from run import Run
    load_torch()
    job = Job(job_name, job_uid)
    run_params = job.run_params(run_name)
    run = Run(job.path, run_params)
    if not run.trained and mode == 'train':
        run.train()
    print(f"[Run {run.run_name}] Processing...")
    job.process_run(run, run.metrics, run.stats)


if __name__ == "__main__":
    load_torch()
    args = parse_args()

    # Load the job
    job = Job(args.job_name, args.uid)

    # Obtain run parameters
    run_params = job.run_params()

    if args.mode == "train":
        # Create job directory and copy job file
        if job.path.exists():
            raise ValueError(f"Job {args.job_name}/{args.uid} already exists.")
        job.path.mkdir(parents=True)
        shutil.copy2(f"jobs/{args.job_name}.py", str(job.path))
    if not job.path.exists():
        raise ValueError(f"Job {args.job_name}/{args.uid} does not exist.")

    # Create and execute runs (train if necessary and process)
    if args.procs > 1:
        mp.set_start_method('spawn')  # This makes sure the different workers can use different seeds
        with mp.Pool(args.procs) as pool:
            pool.starmap(execute_run, ((args.job_name, args.uid, run_name, args.mode) for run_name in run_params.keys()))
    else:
        for run_name in run_params.keys():
            execute_run(args.job_name, args.uid, run_name, args.mode)

    # Analyze runs
    from run import Run
    runs = {name: Run(job.path, params) for name, params in run_params.items()}
    print("Analyzing...")
    job.analyze_runs(job.path, runs)
