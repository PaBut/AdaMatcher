from src.config.default import _CN as cfg

cfg.ADAMATCHER.RESOLUTION = (16, 8, 2)
cfg.ADAMATCHER.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.TRAINER.N_SAMPLES_PER_SUBSET = 200

cfg.TRAINER.WARMUP_TYPE = 'linear'  # 'constant' # 'linear'  # [linear, constant]
cfg.TRAINER.WARMUP_STEP = 4800  # 6400   # 4800
cfg.TRAINER.CANONICAL_LR = 3e-3  # 6e-3

cfg.ADAMATCHER.MATCH_COARSE.SPARSE_SPVS = False

cfg.TRAINER.MSLR_MILESTONES = [
    3,
    6,
    9,
    12,
    17,
    20,
    23,
    26,
    29,
]  # [4,8,12,16,20,23,26,29] # [6, 9, 12, 17, 20, 23, 26, 29] # [3, 6, 9, 12, 17, 20, 23, 26, 29]

cfg.ADAMATCHER.RESOLUTION = (16, 8, 2)  # (32,8,2)
cfg.DATASET.MGDPT_DF = cfg.ADAMATCHER.RESOLUTION[0]
cfg.ADAMATCHER.MATCH_COARSE.CONF_THRESHOLD = 0.5  # 0.3
cfg.ADAMATCHER.MATCH_COARSE.INFERENCE_CONF_THRESHOLD = 0.5
cfg.ADAMATCHER.MATCH_COARSE.CLASS_THRESHOLD = 0.2
cfg.ADAMATCHER.MATCH_COARSE.CLASS_NUM_THRESHOLD = 0.2
cfg.ADAMATCHER.MATCH_COARSE.MAX_O_SCALE = 5.0
cfg.ADAMATCHER.MATCH_COARSE.T_K = -1  # 2048  # -1
cfg.ADAMATCHER.MATCH_COARSE.PATCH_LIMIT_N = 5
