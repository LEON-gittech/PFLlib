import subprocess
import torch
import time

torch.cuda.empty_cache()
datas = [torch.randint(0,100,(100,3000,2000)) for i in range(8)]
for i in range(8):
    datas[i].to(f"cuda:{i}")

def get_gpu_memory_usage():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"]).decode()
    outputs =  output.split("\n")
    print(outputs)
    used = []
    for i in range(8):
        used_memory = float(output.strip().split(', ')[0].strip().split("MiB")[0])
        total_memory = float(output.strip().split(', ')[1].strip().split("MiB")[0])
        used.append(used_memory/total_memory)
    return used

while True:
    print(get_gpu_memory_usage())
    time.sleep(60*5)