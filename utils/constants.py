class Constants:
    BATCH_SIZE = 64  # the number of independent sequences will we process in parallel
    BLOCK_SIZE = 256  # maximum content length for predictions
    LEARNING_RATE = 3e-4
    EVAL_INTERVAL = 500
    EVAL_ITERS = 200
    NUMBER_OF_EMBEDDING_DIMENSIONS = 384
    MAX_ITERS = 5000  # number of iterations to run training loop on
    NUMBER_OF_LAYERS = 6
    NUMBER_OF_HEADS = 6
    DROPOUT = 0.2
