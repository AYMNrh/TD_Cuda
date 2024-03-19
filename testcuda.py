import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def gpu_info():
    # Get GPU device
    device = cuda.Device(0)
    print("GPU Name:", device.name())
    print("GPU Memory:", device.total_memory() / (1024 * 1024), "MB")

def cpu_info():
    # Get CPU information
    import platform
    import psutil

    print("CPU Name:", platform.processor())
    print("Number of Cores:", psutil.cpu_count(logical=False))
    print("Number of Threads (including hyper-threading):", psutil.cpu_count(logical=True))
    print("Total CPU Usage:", psutil.cpu_percent(interval=1), "%")

if __name__ == "__main__":
    gpu_info()
    cpu_info()
