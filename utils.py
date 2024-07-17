import time

def time_run(func,  **kwargs):
    start = time.time()
    func(**kwargs)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
