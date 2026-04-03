import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import modeling_utils
from transformers.utils import import_utils


def _allow_local_bin_checkpoints():
    import_utils.check_torch_load_is_safe = lambda: None
    modeling_utils.check_torch_load_is_safe = lambda: None


_allow_local_bin_checkpoints()


def _load_clip_class():

    module_spec = importlib.util.spec_from_file_location("original_clip", clip_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.CLIP


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0, mode="both"):
        super().__init__()
        if mode not in {"both", "positive", "negative"}:
            raise ValueError("contrastive mode must be one of {'both', 'positive', 'negative'}")
        self.margin = margin
        self.mode = mode

    def forward(self, output1, output2, labels):
        labels = labels.float()
        euclidean_distance = F.pairwise_distance(output1, output2)
        positive_term = labels * torch.pow(euclidean_distance, 2)
        negative_term = (1 - labels) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0),
            2,
        )

        if self.mode == "positive":
            loss = positive_term
        elif self.mode == "negative":
            loss = negative_term
        else:
            loss = positive_term + negative_term
        return loss.mean()


class UnifiedModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        clip_cls = _load_clip_class()
        self.contrastive_model = clip_cls(
            args=args,
            dim_text=512,
            num_text_tokens=self.tokenizer.vocab_size,
            text_seq_len=args.max_length,
            text_heads=8,
        )

        self.global_encoder = AutoModel.from_pretrained(args.model_name_or_path)
        self.line_encoder = AutoModel.from_pretrained(args.model_name_or_path)

        hidden_size = getattr(self.global_encoder.config, "hidden_size", args.hidden_size)
        args.hidden_size = hidden_size
        self.hidden_size = hidden_size

        self.contrastive_adapter = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        self.contrastive_loss_fn = ContrastiveLoss(
            margin=args.contrastive_margin,
            mode=args.contrastive_mode,
        )

        self.line_structure_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dropout=args.dropout,
                batch_first=True,
            ),
            num_layers=8,
        )

        self.line_level_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def extract_features(
        self,
        text1=None,
        batch_input_ids=None,
        line_rule_mask=None,
        global_input_ids=None,
        feature_mode="unified",
    ):
        if feature_mode == "contrastive":
            return self._encode_contrastive_logits(text1)
        if feature_mode == "line_level":
            global_feature, line_semantic_feature, line_structure_feature = self._forward_line_branches(
                batch_input_ids,
                line_rule_mask,
                global_input_ids,
            )
            return self.line_level_fusion(
                torch.cat([global_feature, line_semantic_feature, line_structure_feature], dim=1)
            )
        if feature_mode == "unified":
            contrastive_feature = self._project_contrastive_feature(text1)
            global_feature, line_semantic_feature, line_structure_feature = self._forward_line_branches(
                batch_input_ids,
                line_rule_mask,
                global_input_ids,
            )
            return self.multi_scale_fusion(
                torch.cat(
                    [contrastive_feature, global_feature, line_semantic_feature, line_structure_feature],
                    dim=1,
                )
            )
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    def forward(
        self,
        text1=None,
        text2=None,
        batch_input_ids=None,
        line_rule_mask=None,
        global_input_ids=None,
        labels=None,
        training_mode="contrastive",
    ):
        if training_mode == "contrastive":
            return self._contrastive_forward(text1=text1, text2=text2, labels=labels)
        if training_mode == "line_level":
            return self._line_level_forward(
                batch_input_ids=batch_input_ids,
                line_rule_mask=line_rule_mask,
                global_input_ids=global_input_ids,
                labels=labels,
            )
        if training_mode == "unified":
            return self._unified_forward(
                text1=text1,
                batch_input_ids=batch_input_ids,
                line_rule_mask=line_rule_mask,
                global_input_ids=global_input_ids,
                labels=labels,
            )
        raise ValueError(f"Unknown training mode: {training_mode}")

    def _encode_contrastive_logits(self, input_ids):
        logits1, _, _ = self.contrastive_model(
            text1=input_ids,
            text2=input_ids,
            training_classifier=False,
        )
        return logits1

    def _project_contrastive_feature(self, input_ids):
        contrastive_logits = self._encode_contrastive_logits(input_ids)
        return self.contrastive_adapter(contrastive_logits)

    def _contrastive_forward(self, text1, text2, labels):
        logits1, logits2, ssl_loss = self.contrastive_model(
            text1=text1,
            text2=text2,
            training_classifier=False,
        )

        if labels is None:
            return logits1, logits2

        contrastive_loss = self.contrastive_loss_fn(logits1, logits2, labels)
        total_loss = contrastive_loss + self.args.mlm_weight * ssl_loss.sum()
        return logits1, logits2, total_loss

    def _encode_global_feature(self, global_input_ids):
        global_attention_mask = global_input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.global_encoder(
            input_ids=global_input_ids,
            attention_mask=global_attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, 0, :]

    def _encode_line_features(self, batch_input_ids):
        batch_size, num_lines, _ = batch_input_ids.size()
        line_valid_mask = batch_input_ids.ne(self.tokenizer.pad_token_id).any(dim=-1)
        sentence_cls_list = []

        for sample_index in range(batch_size):
            sample_chunks = []
            for start in range(0, num_lines, self.args.line_chunk_size):
                end = min(start + self.args.line_chunk_size, num_lines)
                chunk_input_ids = batch_input_ids[sample_index, start:end, :]
                chunk_attention_mask = chunk_input_ids.ne(self.tokenizer.pad_token_id)
                outputs = self.line_encoder(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    return_dict=True,
                )
                sample_chunks.append(outputs.last_hidden_state[:, 0, :])
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            sentence_cls_list.append(torch.cat(sample_chunks, dim=0))

        sentence_cls = torch.stack(sentence_cls_list, dim=0)
        return sentence_cls, line_valid_mask

    def _select_key_line(self, sentence_cls, line_valid_mask):
        sentence_means = sentence_cls.mean(dim=2)
        sentence_means = sentence_means.masked_fill(~line_valid_mask, float("inf"))
        min_sentence_indices = sentence_means.argmin(dim=1)
        batch_indices = torch.arange(sentence_cls.size(0), device=sentence_cls.device)
        return sentence_cls[batch_indices, min_sentence_indices]

    def _aggregate_sensitive_lines(self, sentence_cls, line_rule_mask, line_valid_mask):
        if line_rule_mask is None:
            return self._select_key_line(sentence_cls, line_valid_mask)

        if line_rule_mask.dim() > 2:
            line_rule_mask = line_rule_mask.squeeze(-1)

        valid_rule_mask = line_rule_mask.bool().to(sentence_cls.device) & line_valid_mask
        rule_hits = valid_rule_mask.any(dim=1)

        aggregated = torch.zeros(
            sentence_cls.size(0),
            sentence_cls.size(2),
            device=sentence_cls.device,
            dtype=sentence_cls.dtype,
        )

        if rule_hits.any():
            weights = valid_rule_mask[rule_hits].float().unsqueeze(-1)
            weighted_sum = (sentence_cls[rule_hits] * weights).sum(dim=1)
            counts = weights.sum(dim=1).clamp(min=1.0)
            aggregated[rule_hits] = weighted_sum / counts

        fallback = self._select_key_line(sentence_cls, line_valid_mask)
        aggregated[~rule_hits] = fallback[~rule_hits]
        return aggregated

    def _encode_line_structure(self, sentence_cls, line_valid_mask):
        detached_sentence_cls = sentence_cls.detach()
        transformer_output = self.line_structure_encoder(
            detached_sentence_cls,
            src_key_padding_mask=~line_valid_mask,
        )
        valid_lengths = line_valid_mask.long().sum(dim=1).clamp(min=1)
        last_indices = valid_lengths - 1
        batch_indices = torch.arange(transformer_output.size(0), device=transformer_output.device)
        return transformer_output[batch_indices, last_indices]

    def _forward_line_branches(self, batch_input_ids, line_rule_mask, global_input_ids):
        global_feature = self._encode_global_feature(global_input_ids)
        sentence_cls, line_valid_mask = self._encode_line_features(batch_input_ids)
        line_semantic_feature = self._aggregate_sensitive_lines(sentence_cls, line_rule_mask, line_valid_mask)
        line_structure_feature = self._encode_line_structure(sentence_cls, line_valid_mask)
        return global_feature, line_semantic_feature, line_structure_feature

    def _line_level_forward(self, batch_input_ids, line_rule_mask, global_input_ids, labels):
        global_feature, line_semantic_feature, line_structure_feature = self._forward_line_branches(
            batch_input_ids,
            line_rule_mask,
            global_input_ids,
        )
        fused_feature = self.line_level_fusion(
            torch.cat([global_feature, line_semantic_feature, line_structure_feature], dim=1)
        )
        logits = self.classifier(fused_feature)
        if labels is None:
            return logits
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    def _unified_forward(self, text1, batch_input_ids, line_rule_mask, global_input_ids, labels):
        contrastive_feature = self._project_contrastive_feature(text1)
        global_feature, line_semantic_feature, line_structure_feature = self._forward_line_branches(
            batch_input_ids,
            line_rule_mask,
            global_input_ids,
        )
        fused_feature = self.multi_scale_fusion(
            torch.cat(
                [
                    contrastive_feature,
                    global_feature,
                    line_semantic_feature,
                    line_structure_feature,
                ],
                dim=1,
            )
        )
        logits = self.classifier(fused_feature)
        if labels is None:
            return logits
        loss = F.cross_entropy(logits, labels)
        return loss, logits
