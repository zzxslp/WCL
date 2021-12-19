import torch
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

def draw_loss_figure(all_logs, save_dir):
    train_loss = [log['train_loss'] for log in all_logs]
    val_BLEU_1 = [log['val_BLEU_1'] for log in all_logs]
    val_BLEU_4 = [log['val_BLEU_4'] for log in all_logs]
    val_METEOR = [log['val_METEOR'] for log in all_logs]
    val_ROUGE_L = [log['val_ROUGE_L'] for log in all_logs]
    x = np.arange(len(all_logs))
    plt.plot(x, train_loss)
    plt.plot(x, val_BLEU_1)
    plt.plot(x, val_BLEU_4)
    plt.plot(x, val_METEOR)
    plt.plot(x, val_ROUGE_L)

    plt.legend(['train_loss', 'val_BLEU_1', 'val_BLEU_4', 'val_METEOR', 'val_ROUGE_L'], loc='upper right')
    plt.savefig(os.path.join(save_dir, 'loss.png'))

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x

def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

