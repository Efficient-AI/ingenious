import argparse
import datetime
import time
import logging
import math
import os
import sys
import random
import datasets
import torch
from torch.optim import AdamW
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.utils import broadcast_object_list
from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertForPreTraining,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler
)
from transformers.utils.versions import require_version
from cords.selectionstrategies.SL import SubmodStrategy
import pickle
from accelerate import InitProcessGroupKwargs
import faiss

logger=get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

