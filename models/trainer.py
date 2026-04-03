import copy
import gc
import logging
import os
from itertools import cycle

import numpy as np
import torch
from tqdm import tqdm

try:
    from utils.visualization import save_embedding_plot
except ImportError:
    from ..utils.visualization import save_embedding_plot


logger = logging.getLogger(__name__)


def compute_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    denominator = max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denominator

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


class MultiStageTrainer:
    def __init__(self, model, args):
        self.model = model.to(args.device)
        self.args = args
        self.device = args.device
        self.best_state_dict = None
        self.best_f1 = -1.0

    def _build_optimizer(self, params):
        return torch.optim.AdamW(
            params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_contrastive(self, train_loader, pos_loader):
        if train_loader is None or pos_loader is None:
            logger.info("跳过对比学习阶段：缺少训练数据或正样本数据。")
            return

        optimizer = self._build_optimizer(self.model.parameters())
        self.model.train()

        for epoch in range(self.args.contrastive_epochs):
            total_loss = 0.0
            progress = tqdm(zip(train_loader, cycle(pos_loader)), total=len(train_loader), desc=f"contrastive-{epoch + 1}")
            for train_batch, pos_batch in progress:
                input_ids = train_batch["input_ids"].to(self.device)
                labels = train_batch["labels"].to(self.device)
                pos_input_ids = pos_batch["input_ids"].to(self.device)

                batch_size = min(input_ids.size(0), pos_input_ids.size(0))
                input_ids = input_ids[:batch_size]
                labels = labels[:batch_size]
                pos_input_ids = pos_input_ids[:batch_size]

                pair_labels = labels.float()
                optimizer.zero_grad()
                _, _, loss = self.model(
                    text1=input_ids,
                    text2=pos_input_ids,
                    labels=pair_labels,
                    training_mode="contrastive",
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            logger.info("对比学习 Epoch %d/%d loss=%.4f", epoch + 1, self.args.contrastive_epochs, total_loss / max(len(train_loader), 1))
            self._cleanup_memory()

    def train_line_level(self, train_loader, eval_loader=None):
        if train_loader is None:
            logger.info("跳过行级语义学习阶段：缺少训练数据。")
            return

        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not any(key in name for key in ["clear_contrastive_model", "contrastive_adapter"])

        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = self._build_optimizer(trainable_params)

        for epoch in range(self.args.line_level_epochs):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(train_loader, total=len(train_loader), desc=f"line-level-{epoch + 1}")
            for batch in progress:
                optimizer.zero_grad()
                loss, _ = self.model(
                    batch_input_ids=batch["line_input_ids"].to(self.device),
                    line_rule_mask=batch["line_rule_mask"].to(self.device),
                    global_input_ids=batch["global_input_ids"].to(self.device),
                    labels=batch["labels"].to(self.device),
                    training_mode="line_level",
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, self.args.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            logger.info("行级阶段 Epoch %d/%d loss=%.4f", epoch + 1, self.args.line_level_epochs, total_loss / max(len(train_loader), 1))
            if eval_loader is not None:
                metrics = self.evaluate(eval_loader, mode="line_level")
                logger.info("行级验证指标: %s", metrics)
            self._cleanup_memory()

        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def train_unified(self, train_loader, eval_loader=None):
        if train_loader is None:
            logger.info("跳过统一微调阶段：缺少训练数据。")
            return

        optimizer = self._build_optimizer(self.model.parameters())

        for epoch in range(self.args.unified_epochs):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(train_loader, total=len(train_loader), desc=f"unified-{epoch + 1}")
            for batch in progress:
                optimizer.zero_grad()
                loss, _ = self.model(
                    text1=batch["contrastive_input_ids"].to(self.device),
                    batch_input_ids=batch["line_input_ids"].to(self.device),
                    line_rule_mask=batch["line_rule_mask"].to(self.device),
                    global_input_ids=batch["global_input_ids"].to(self.device),
                    labels=batch["labels"].to(self.device),
                    training_mode="unified",
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            logger.info("统一微调 Epoch %d/%d loss=%.4f", epoch + 1, self.args.unified_epochs, total_loss / max(len(train_loader), 1))
            if eval_loader is not None:
                metrics = self.evaluate(eval_loader, mode="unified")
                logger.info("统一阶段验证指标: %s", metrics)
                if metrics["f1"] > self.best_f1:
                    self.best_f1 = metrics["f1"]
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
            self._cleanup_memory()

    def evaluate(self, data_loader, mode="unified"):
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader), desc=f"eval-{mode}"):
                if mode == "line_level":
                    logits = self.model(
                        batch_input_ids=batch["line_input_ids"].to(self.device),
                        line_rule_mask=batch["line_rule_mask"].to(self.device),
                        global_input_ids=batch["global_input_ids"].to(self.device),
                        training_mode="line_level",
                    )
                else:
                    logits = self.model(
                        text1=batch["contrastive_input_ids"].to(self.device),
                        batch_input_ids=batch["line_input_ids"].to(self.device),
                        line_rule_mask=batch["line_rule_mask"].to(self.device),
                        global_input_ids=batch["global_input_ids"].to(self.device),
                        training_mode="unified",
                    )

                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(batch["labels"].cpu().numpy().tolist())

        return compute_binary_metrics(all_labels, all_predictions)

    def save_best(self, save_path):
        state_dict = self.best_state_dict if self.best_state_dict is not None else self.model.state_dict()
        torch.save(state_dict, save_path)
        logger.info("模型已保存到 %s", save_path)

    def save_stage_checkpoint(self, file_name):
        save_path = os.path.join(self.args.output_dir, file_name)
        torch.save(self.model.state_dict(), save_path)
        logger.info("阶段模型已保存到 %s", save_path)

    def visualize_embeddings(self, data_loader, stage_name, feature_mode):
        if data_loader is None or self.args.disable_visualization:
            return None

        self.model.eval()
        all_features = []
        all_labels = []
        seen_samples = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader), desc=f"visualize-{stage_name}"):
                features = self.model.extract_features(
                    text1=batch.get("contrastive_input_ids", batch.get("input_ids")).to(self.device),
                    batch_input_ids=batch.get("line_input_ids").to(self.device) if batch.get("line_input_ids") is not None else None,
                    line_rule_mask=batch.get("line_rule_mask").to(self.device) if batch.get("line_rule_mask") is not None else None,
                    global_input_ids=batch.get("global_input_ids").to(self.device) if batch.get("global_input_ids") is not None else None,
                    feature_mode=feature_mode,
                )
                labels = batch["labels"]

                all_features.append(features.detach().cpu())
                all_labels.append(labels.detach().cpu())
                seen_samples += labels.size(0)

                if self.args.visualization_max_samples is not None and seen_samples >= self.args.visualization_max_samples:
                    break

        feature_tensor = torch.cat(all_features, dim=0)
        label_tensor = torch.cat(all_labels, dim=0)
        if self.args.visualization_max_samples is not None:
            feature_tensor = feature_tensor[: self.args.visualization_max_samples]
            label_tensor = label_tensor[: self.args.visualization_max_samples]

        save_path = os.path.join(self.args.output_dir, f"{stage_name}_test_pca.svg")
        save_embedding_plot(
            feature_tensor.numpy(),
            label_tensor.numpy(),
            save_path=save_path,
            title=f"Test Set PCA - {stage_name}",
        )
        logger.info("可视化已保存到 %s", save_path)
        return save_path

    def train_all_stages(self, contrastive_loader, positive_loader, train_loader, val_loader, test_loader):
        self.visualize_embeddings(
            test_loader,
            stage_name="before_training",
            feature_mode=self.args.visualization_mode_before,
        )

        if self.args.enable_contrastive:
            self.train_contrastive(contrastive_loader, positive_loader)
            self.save_stage_checkpoint("contrastive_stage.pt")
            self.visualize_embeddings(
                test_loader,
                stage_name="after_contrastive",
                feature_mode=self.args.visualization_mode_after_contrastive,
            )
        if self.args.enable_line_level:
            self.train_line_level(train_loader, val_loader)
        if self.args.enable_unified:
            self.train_unified(train_loader, val_loader)

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        target_loader = test_loader if test_loader is not None else val_loader
        if target_loader is None:
            return {}
        final_mode = "unified" if self.args.enable_unified else "line_level"
        final_metrics = self.evaluate(target_loader, mode=final_mode)
        self.visualize_embeddings(
            target_loader,
            stage_name="after_training",
            feature_mode=self.args.visualization_mode_final,
        )
        return final_metrics
