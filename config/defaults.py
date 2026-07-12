# Circuit
N_QUBITS = 20
N_LAYERS = 4
N_FEATURES = 20

# Training
BATCH_SIZE = 512
LEARNING_RATE = 0.005
PATIENCE = 10

# LR scheduler
LR_WARMUP_EPOCHS = 3          # linear warmup from LR/10 to LR
LR_SCHEDULER_PATIENCE = 5     # epochs without F1 improvement before halving LR
LR_SCHEDULER_FACTOR = 0.5     # multiply LR by this on plateau
LR_MIN = 1e-5                 # floor for LR reduction

# Focal loss
FOCAL_LOSS_GAMMA = 2.0

# Gradient clipping
MAX_GRAD_NORM = 0.75

# Backend
DEFAULT_BACKEND = "lightning.gpu"

# Column removal — nPrint feature indices that encode network-specific
# identifiers rather than OS behavior. Removing them prevents the classifier
# from memorizing which host/network a packet came from. Index ranges are
# kept identical to OsirisML (the XGBoost baseline) so results are comparable.
# A few boundary indices appear in two ranges; pandas dedups on drop.
COLUMNS_TO_REMOVE = (
    [0]                        # src_ip string column prepended by the capture tooling
    + list(range(97, 130))     # IPv4 source address bits
    + list(range(130, 161))    # IPv4 destination address bits
    + list(range(33, 49))      # IPv4 identification field bits
    + list(range(480, 497))    # TCP source/destination port bits
    + list(range(496, 513))    # TCP source/destination port bits (cont.)
    + list(range(512, 545))    # TCP sequence number bits
    + list(range(544, 577))    # TCP acknowledgment number bits
)
