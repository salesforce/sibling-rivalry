if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
from dist_train.utils.experiment_bookend import open_experiment
from dist_train.workers import synchronous_worker


if __name__ == '__main__':
    # Interpret the arguments. Load the shared model/optimizer. Fetch the config file.
    model, _, config, args = open_experiment(apply_time_machine=True)

    print(' ', flush=True)
    model.reset()
    print(' ', flush=True)

    # Create a group of workers
    print('Launching the individual workers...', flush=True)
    processes = []
    for rank in range(args.N):
        # The workers perform roll-outs and synchronize gradients
        p = mp.Process(target=synchronous_worker, args=(int(rank), config, args))
        p.start()
        time.sleep(0.25)
        processes.append(p)

    for p in processes:
        p.join()
