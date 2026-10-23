import torch
import pickle
import glob

label = "gait-conditioned-agility/pretrain-go2/train"
dirs = glob.glob(f"../../runs/{label}/*")
logdir = sorted(dirs)[0]

# transfer
with open(logdir + "/parameters.pkl", "rb") as file:
    pkl_cfg = pickle.load(file)
    # We use torch.is_tensor(v) to check whether each value v in the dict is a PyTorch tensor.
    # If it is, we call .cpu() to move it to the CPU; if not, we keep the original value.
    # This way only actual tensors are moved; other value types (ints, strings, etc.) stay unchanged.
    pkl_cfg_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in pkl_cfg.items()}
    print("Transfer Succeed ! !")

# save transferred .pkl file
with open(logdir + "/parameters_cpu.pkl", "wb") as file:
    pickle.dump(pkl_cfg_cpu, file)
    print("Transferred Pickle File has been saved as parameters_cpu.pkl")
