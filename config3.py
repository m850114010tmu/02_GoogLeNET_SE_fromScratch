
import os
import random
import numpy as np
import torch
from pathlib import Path

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

def seed_everything(seed: int = SEED):
    """Lock every RNG source so a re-run reproduces identical results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False         
    os.environ["PYTHONHASHSEED"] = str(seed)

# ─── Paths (Ex3!) ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"datapath")
DATA_DIR    = BASE_DIR / "vir_data"
TRAIN_DIR   = DATA_DIR / "train"
TEST_DIR    = DATA_DIR / "test1_vir"
OUTPUT_DIR  = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Classes ──────────────────
CLASS_NAMES  = ["class 1", "class 2", "class 3"]
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ─── Training hyperparameters ─────────────────────────────────────────────────
IMAGE_SIZE       = 224      
BATCH_SIZE       = 32       
GRAD_ACCUM_STEPS = 1        
EPOCHS           = 60       
WARMUP_EPOCHS    = 5        
LEARNING_RATE    = 3e-4     
WEIGHT_DECAY     = 1e-4     
PATIENCE         = 12       
VAL_SPLIT        = 0.2      
LABEL_SMOOTHING  = 0.1      
DROPOUT          = 0.4      

# ─── Architecture feature flags ───────────────────────────────────────────────
USE_SE        = True        
SE_REDUCTION  = 16          
USE_AUX       = True        
AUX_WEIGHT    = 0.3         
COMPARE_SE    = True        

# ─── Mixup ──────────────────────────────────────────
USE_MIXUP   = False         
MIXUP_ALPHA = 0.2

# ─── Augmentation strength ─────────────────────
COLOR_JITTER  = (0.3, 0.3, 0.3, 0.10)   
CHANNEL_SCALE = 0.15                    
NOISE_STD     = 0.04                    

# ─── DataLoader workers ─────────
NUM_WORKERS = 2

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Output file paths  ──────────────────────────────
MODEL_SAVE_PATH      = OUTPUT_DIR / "report3.pth"  
MODEL_SAVE_PATH_ROOT = BASE_DIR  / "report3.pth"    
HISTORY_SAVE_PATH    = OUTPUT_DIR / "training_history.csv"
UTIL_SAVE_PATH       = BASE_DIR  / "util3.txt"
STATS_PATH           = OUTPUT_DIR / "train_stats.json"
