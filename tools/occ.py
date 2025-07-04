import torch
import time
import pynvml
gpu_list = [0,1, 2, 3, 4, 5, 6,7]
# gpu_list = [0]
done_gpu_list = []
# gpu_list = [4, 5, 6, 7]
sleep_len = 1000
mode = 0
pynvml.nvmlInit()
occ_ratio = 0.8
while(len(done_gpu_list) < len(gpu_list)):
    for rank in gpu_list:
        if mode != 1:
            time.sleep(1)
        if rank in done_gpu_list:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(rank, meminfo.used)
        
        remain_memory = meminfo.total - meminfo.used
        use_ratio = meminfo.used / meminfo.total
        remain_memory_GB = remain_memory / 1024 / 1024 / 1024
        if mode == 1:
            threshold = 1.4
        else:
            threshold = 0.4
        print("rank: ", rank, "remain_memory: ", remain_memory_GB, "use_ratio: ", use_ratio, "threshold: ", threshold, 'occ_ratio: ', occ_ratio)

        if use_ratio < threshold:
            done_gpu_list.append(rank)
            print("occupy gpu: ", rank)
            # print("rank: ", rank, "remain_memory: ", remain_memory_GB, "use_ratio: ", use_ratio)
            if mode == 1:
                dummy_tensor = torch.empty(int(remain_memory * 0.5), dtype=torch.uint8, device=rank) # 创建一个占据目标显存的张量
            else:
                dummy_tensor = torch.empty(int(remain_memory * occ_ratio), dtype=torch.uint8, device=rank)
#     dummy_tensor = torch.empty(target_memory, dtype=torch.uint8, device=device) # 创建一个占据目标显存的张量
for i in range(sleep_len):
    time.sleep(1000)
    print(i)


