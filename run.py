#!/usr/bin/env python3

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import get_args, save_args
from models import MultiStageTrainer, UnifiedClearCSLSModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(CURRENT_DIR / "training.log")],
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = get_args()
    set_seed(args.seed)

    from utils.data_utils import build_dataloaders

    save_args(args, os.path.join(args.output_dir, "config.json"))
    logger.info("开始运行统一框架: Clear 对比学习 + CSLS 行级建模")
    logger.info("配置: %s", json.dumps(vars(args), ensure_ascii=False))

    model = UnifiedClearCSLSModel(args)
    tokenizer = model.tokenizer
    contrastive_loader, positive_loader, train_loader, val_loader, test_loader = build_dataloaders(args, tokenizer)

    trainer = MultiStageTrainer(model, args)
    final_metrics = trainer.train_all_stages(
        contrastive_loader=contrastive_loader,
        positive_loader=positive_loader,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    trainer.save_best(os.path.join(args.output_dir, "best_model.pt"))
    pd.DataFrame([final_metrics]).to_csv(os.path.join(args.output_dir, "final_metrics.csv"), index=False)

    logger.info("最终指标: %s", final_metrics)


if __name__ == "__main__":
    main()
