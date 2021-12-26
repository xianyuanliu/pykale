"""
Default configurations for action recognition
"""

import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "J:/Datasets/EgoAction/"  # "/shared/tale2/Shared"
_C.DATASET.NAME = "EPIC"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN"]
_C.DATASET.TRAINLIST = "epic_D1_train.pkl"
_C.DATASET.TESTLIST = "epic_D1_test.pkl"
_C.DATASET.IMAGE_MODALITY = "rgb"  # mode options=["rgb", "flow", "joint"]
_C.DATASET.FRAMES_PER_SEGMENT = 16
_C.DATASET.NUM_REPEAT = 5  # 10
_C.DATASET.VALID_RATIO = 0.1
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.BASE_LR = 0.01  # learning rate
# _C.SOLVER.MOMENTUM = 0.9
# _C.SOLVER.WEIGHT_DECAY = 0.0005  # 1e-4
# _C.SOLVER.NESTEROV = True

_C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 30
# _C.SOLVER.WARMUP = True
_C.SOLVER.TRAIN_BATCH_SIZE = 16  # 150
_C.SOLVER.TEST_BATCH_SIZE = 32

# ---------------------------------------------------------------------------- #
# Network configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.METHOD = "i3d"  # options=["r3d_18", "r2plus1d_18", "mc3_18", "i3d", "c3d"]
_C.MODEL.ATTENTION = "None"  # options=["None", "SELayer"]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.VERBOSE = False  # To discuss, for HPC jobs
_C.OUTPUT.FAST_DEV_RUN = False  # True for debug
_C.OUTPUT.PB_FRESH = 0  # 0 # 50 # 0 to disable  ; MAYBE make it a command line option
_C.OUTPUT.TB_DIR = os.path.join("tb_logs", _C.DATASET.NAME)


def get_cfg_defaults():
    return _C.clone()
