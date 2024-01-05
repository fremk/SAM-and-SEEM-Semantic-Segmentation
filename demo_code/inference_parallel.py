import subprocess
import multiprocessing
import os 

def run_inference_with_gpu(args, gpu_number):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
        subprocess.run(['python3', 'inference_total.py'] + args, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running inference_total.py: {e}")

if __name__ == "__main__":
    # Uses all 4 GPUs of titan2. Modify at will. 
    # First argument must be the batch number which just goes from 0 till the total nb of batches-1 for each process.
    # Second argument is the total number of batches you want to divide the full images batch on. And finally you have the GPU number for each of the processes, 
    # usually 2 processes per GPU since each process uses ~10GB. 
    runs = [
        (["--batch", "0", "--nb_batches", "8"], "0"),
        (["--batch", "1", "--nb_batches", "8"], "0"),
        (["--batch", "2", "--nb_batches", "8"], "1"),
        (["--batch", "3", "--nb_batches", "8"], "1"),
        (["--batch", "4", "--nb_batches", "8"], "2"),
        (["--batch", "5", "--nb_batches", "8"], "2"),
        (["--batch", "6", "--nb_batches", "8"], "3"),
        (["--batch", "7", "--nb_batches", "8"], "3"),
    ]

    with multiprocessing.Pool(processes=len(runs)) as pool:
        pool.starmap(run_inference_with_gpu, runs)