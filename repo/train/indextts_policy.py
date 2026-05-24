"""IndexTTS UnifiedVoice policy wrapper for GRPO training.

Responsibilities:
- Load SFT UnifiedVoice GPT checkpoint via IndexTTS2.
- Prepare speaker/emotion conditioning from a prompt wav.
- Sample group_size semantic-code sequences (rollout) and return
  codes + old_logprobs (no gradient).
- Compute new logprobs via teacher-forcing forward pass (with gradient).
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# VA → alpha MLP
# ---------------------------------------------------------------------------

class VAAlphaMLP(nn.Module):
    """Mean VA (0-1 scale, 2D) → scalar alpha ∈ (0,1) via small MLP."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, va_01: torch.Tensor) -> torch.Tensor:
        """va_01: [2] → alpha: scalar tensor"""
        return self.net(va_01.float()).squeeze(-1)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Conditioning:
    """Pre-computed speaker/emotion conditioning tensors."""
    spk_cond_emb: torch.Tensor           # [1, T_frames, dim] w2v2bert features
    emovec: torch.Tensor                  # [1, dim] merged emotion vector
    # S2mel decode extras (set by prepare_conditioning)
    style: torch.Tensor                   # [1, 192] CAMPPlus style
    prompt_condition: torch.Tensor        # [1, T_ref_upsampled, dim]
    ref_mel: torch.Tensor                 # [1, n_mels, T_ref]
    # Filled lazily by rollout()
    speech_conditioning_latent: Optional[torch.Tensor] = field(default=None)
    # For MLP alpha-shift gradient in compute_logprobs
    base_emovec: Optional[torch.Tensor] = field(default=None)       # [1, dim] speaker-only
    emo_mat: Optional[torch.Tensor] = field(default=None)           # [n_emo, dim] sampled rows
    va_01: Optional[tuple] = field(default=None)                    # mean VA 0-1 scale
    base_emotion_vector: Optional[list] = field(default=None)       # unscaled one-hot


@dataclass
class RolloutResult:
    """Output of a single rollout call (group_size sequences)."""
    codes: list[torch.Tensor]           # K tensors, each shape [T_k]
    old_logprobs: list[torch.Tensor]    # K tensors, each shape [T_k]
    speech_conditioning_latent: torch.Tensor  # [1, 32, dim] perceiver output
    emovec: torch.Tensor                      # [1, dim]
    text_tokens: torch.Tensor                 # [1, L] int32
    use_speed: torch.Tensor                   # [1] long
    # For MLP alpha-shift gradient in compute_logprobs
    base_emovec: Optional[torch.Tensor] = field(default=None)
    emo_mat: Optional[torch.Tensor] = field(default=None)
    va_01: Optional[tuple] = field(default=None)
    base_emotion_vector: Optional[list] = field(default=None)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class IndexTTSPolicy:
    def __init__(
        self,
        cfg_path: str,
        model_dir: str,
        gpt_checkpoint_path: str,
        bpe_model_path: str,
        device: str = "cuda:0",
    ) -> None:
        _ensure_index_tts_on_path()
        from indextts.infer_v2_modded import IndexTTS2

        self.tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            gpt_checkpoint_path=gpt_checkpoint_path,
            bpe_model_path=bpe_model_path,
            device=device,
            use_fp16=False,
            use_deepspeed=False,
        )
        self.device = device
        self.gpt = self.tts.gpt
        self.stop_mel_token = self.tts.stop_mel_token
        self.va_alpha_mlp = VAAlphaMLP(hidden=16).to(device)

    # ------------------------------------------------------------------
    # Conditioning preparation
    # ------------------------------------------------------------------

    def prepare_conditioning(
        self,
        spk_audio_path: str,
        emotion_vector: Optional[list[float]] = None,
        va_01: Optional[tuple] = None,
    ) -> Conditioning:
        """Compute all speaker/emotion tensors needed for rollout and decode."""
        import librosa
        import torchaudio

        tts = self.tts
        gpt = self.gpt
        device = self.device

        audio, sr = librosa.load(spk_audio_path)
        audio_t = torch.tensor(audio).unsqueeze(0)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio_t)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_t)

        inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        saved_emo_mat: Optional[torch.Tensor] = None

        with torch.no_grad():
            spk_cond_emb = tts.get_emb(input_features, attention_mask)  # [1, T, dim]

            # S2mel prompt conditioning
            _, S_ref = tts.semantic_codec.quantize(spk_cond_emb)
            ref_mel = tts.mel_fn(audio_22k.to(device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(device)
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(device), num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = tts.campplus_model(feat.unsqueeze(0))                   # [1, 192]
            prompt_condition = tts.s2mel.models["length_regulator"](
                S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
            )[0]

            # Speaker-only base emotion vector
            cond_len = torch.tensor([spk_cond_emb.shape[-1]], device=device)
            base_emovec = gpt.merge_emovec(
                spk_cond_emb, spk_cond_emb, cond_len, cond_len, alpha=1.0
            )  # [1, dim]
            emovec = base_emovec

            if emotion_vector is not None and any(v > 0 for v in emotion_vector):
                # Compute alpha from VA values (use item() to get a plain float for rollout)
                if va_01 is not None:
                    va_t = torch.tensor(list(va_01), device=device)
                    alpha = float(self.va_alpha_mlp(va_t).item())
                else:
                    alpha = 1.0

                scaled_ev = [v * alpha for v in emotion_vector]
                weight_vector = torch.tensor(scaled_ev, device=device)

                rand_idx = [random.randint(0, x - 1) for x in tts.emo_num]
                emo_mats = [
                    tts.emo_matrix[i][rand_idx[i]].unsqueeze(0)
                    for i in range(len(rand_idx))
                ]
                saved_emo_mat = torch.cat(emo_mats, 0)  # [n_emo, dim]

                emovec_mat = torch.sum(
                    weight_vector.unsqueeze(1) * saved_emo_mat, 0
                ).unsqueeze(0)
                emovec = emovec_mat + (1.0 - torch.sum(weight_vector)) * base_emovec

        return Conditioning(
            spk_cond_emb=spk_cond_emb,
            emovec=emovec,
            style=style,
            prompt_condition=prompt_condition,
            ref_mel=ref_mel,
            base_emovec=base_emovec,
            emo_mat=saved_emo_mat,
            va_01=va_01,
            base_emotion_vector=emotion_vector,
        )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        text: str,
        cond: Conditioning,
        group_size: int = 4,
        max_mel_tokens: int = 512,
        top_k: int = 30,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> RolloutResult:
        """Sample group_size sequences; return codes and per-token old logprobs."""
        tts = self.tts
        gpt = self.gpt
        device = self.device

        text_ids = tts.tokenizer.convert_tokens_to_ids(
            tts.tokenizer.split_segments(
                tts.tokenizer.tokenize(text, skip_normalizer=True),
                max_text_tokens_per_segment=120,
            )[0]
        )
        text_tokens = torch.tensor(text_ids, dtype=torch.int32, device=device).unsqueeze(0)

        spk_cond_emb = cond.spk_cond_emb
        emovec = cond.emovec
        cond_len = torch.tensor([spk_cond_emb.shape[-1]], device=device)

        output, speech_conditioning_latent = gpt.inference_speech(
            spk_cond_emb,
            text_tokens,
            spk_cond_emb,            # emo_speech_condition placeholder
            cond_lengths=cond_len,
            emo_cond_lengths=cond_len,
            emo_vec=emovec,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=group_size,
            max_generate_length=max_mel_tokens,
            return_dict_in_generate=True,
            output_logits=True,
        )

        sequences = output.sequences  # [K, T_gen]
        logits_tuple = output.logits  # tuple of T_gen tensors, each [K, vocab]

        if logits_tuple:
            logits_stack = torch.stack(logits_tuple, dim=1)  # [K, T_gen, vocab]
            log_probs_all = F.log_softmax(logits_stack, dim=-1)
            gathered = log_probs_all.gather(
                2, sequences.long().unsqueeze(-1)
            ).squeeze(-1)  # [K, T_gen]
        else:
            gathered = torch.zeros(sequences.shape[0], sequences.shape[1], device=device)

        codes_list: list[torch.Tensor] = []
        logprobs_list: list[torch.Tensor] = []
        use_speed = torch.zeros(1, device=device).long()

        for k in range(group_size):
            seq_k = sequences[k]
            lp_k = gathered[k]
            stop_idxs = (seq_k == self.stop_mel_token).nonzero(as_tuple=False)
            end = int(stop_idxs[0].item()) if len(stop_idxs) > 0 else len(seq_k)
            codes_list.append(seq_k[:end].clone())
            logprobs_list.append(lp_k[:end].clone())

        cond.speech_conditioning_latent = speech_conditioning_latent

        return RolloutResult(
            codes=codes_list,
            old_logprobs=logprobs_list,
            speech_conditioning_latent=speech_conditioning_latent,
            emovec=emovec,
            text_tokens=text_tokens,
            use_speed=use_speed,
            base_emovec=cond.base_emovec,
            emo_mat=cond.emo_mat,
            va_01=cond.va_01,
            base_emotion_vector=cond.base_emotion_vector,
        )

    # ------------------------------------------------------------------
    # Logprob computation (with gradient)
    # ------------------------------------------------------------------

    def snapshot_reference(self) -> None:
        """Freeze a deep copy of gpt as the reference model for KL regularization."""
        import copy
        self._ref_gpt = copy.deepcopy(self.gpt)
        self._ref_gpt.eval()
        for p in self._ref_gpt.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute_ref_logprobs(self, rollout: RolloutResult, k: int) -> torch.Tensor:
        """Reference logprobs under the frozen initial model (no gradient)."""
        if not hasattr(self, "_ref_gpt"):
            raise RuntimeError("Call snapshot_reference() before compute_ref_logprobs()")

        gpt = self._ref_gpt
        device = self.device

        codes = rollout.codes[k].long().unsqueeze(0)
        speech_cond_latent = rollout.speech_conditioning_latent
        emovec = rollout.emovec
        text_tokens = rollout.text_tokens.long()
        use_speed = rollout.use_speed
        T = codes.shape[1]

        d_emb = gpt.speed_emb(torch.zeros_like(use_speed))
        d_emb_half = gpt.speed_emb(torch.ones_like(use_speed))
        conds = torch.cat(
            (
                speech_cond_latent + emovec.unsqueeze(1),
                d_emb_half.unsqueeze(1),
                d_emb.unsqueeze(1),
            ),
            dim=1,
        )

        text_len = torch.tensor([text_tokens.shape[1]], device=device)
        text_pad = gpt.set_text_padding(text_tokens.clone(), text_len)
        text_pad = F.pad(text_pad, (0, 1), value=gpt.stop_text_token)
        text_in, _ = gpt.build_aligned_inputs_and_targets(
            text_pad, gpt.start_text_token, gpt.stop_text_token
        )
        text_emb = gpt.text_embedding(text_in) + gpt.text_pos_embedding(text_in)

        mel_pad = F.pad(codes, (0, 1), value=gpt.stop_mel_token)
        mel_in, _ = gpt.build_aligned_inputs_and_targets(
            mel_pad, gpt.start_mel_token, gpt.stop_mel_token
        )
        mel_emb = gpt.mel_embedding(mel_in) + gpt.mel_pos_embedding(mel_in)

        emb = torch.cat([conds, text_emb, mel_emb], dim=1)
        gpt_out = gpt.gpt(inputs_embeds=emb, return_dict=True)
        enc = gpt.final_norm(gpt_out.last_hidden_state[:, conds.shape[1]:])

        mel_enc = enc[:, -mel_emb.shape[1]:]
        mel_logits = gpt.mel_head(mel_enc)

        log_probs = F.log_softmax(mel_logits, dim=-1)
        lp = log_probs[:, :T].gather(2, codes.unsqueeze(-1)).squeeze(-1)
        return lp.squeeze(0)

    def compute_logprobs(self, rollout: RolloutResult, k: int) -> torch.Tensor:
        """Teacher-forcing forward pass → logprobs for codes[k].

        Gradient flows through mel_embedding, GPT layers, and mel_head.
        Returns shape [T] matching rollout.codes[k].
        """
        gpt = self.gpt
        device = self.device

        codes = rollout.codes[k].long().unsqueeze(0)      # [1, T]
        speech_cond_latent = rollout.speech_conditioning_latent  # [1, 32, dim]
        emovec = rollout.emovec                            # [1, dim]
        text_tokens = rollout.text_tokens.long()           # [1, L]
        use_speed = rollout.use_speed                      # [1]
        T = codes.shape[1]

        # Recompute alpha-shifted emovec with gradient so MLP is trained end-to-end
        if (
            rollout.va_01 is not None
            and rollout.base_emotion_vector is not None
            and rollout.emo_mat is not None
            and rollout.base_emovec is not None
        ):
            va_t = torch.tensor(list(rollout.va_01), device=device)
            alpha = self.va_alpha_mlp(va_t)                        # scalar, has grad
            weight_vector = (
                torch.tensor(rollout.base_emotion_vector, device=device) * alpha
            )                                                       # [n_emo], has grad
            emo_mat = rollout.emo_mat.to(device)                   # [n_emo, dim]
            emovec_mat = torch.sum(
                weight_vector.unsqueeze(1) * emo_mat, dim=0
            ).unsqueeze(0)                                          # [1, dim]
            base_ev = rollout.base_emovec.to(device).detach()      # [1, dim]
            emovec = emovec_mat + (1.0 - weight_vector.sum()) * base_ev

        # Conditioning (mirrors UnifiedVoice.forward)
        d_emb = gpt.speed_emb(torch.zeros_like(use_speed))       # [1, dim]
        d_emb_half = gpt.speed_emb(torch.ones_like(use_speed))   # [1, dim]
        conds = torch.cat(
            (
                speech_cond_latent + emovec.unsqueeze(1),
                d_emb_half.unsqueeze(1),
                d_emb.unsqueeze(1),
            ),
            dim=1,
        )  # [1, 32+2, dim]

        # Text embeddings
        text_len = torch.tensor([text_tokens.shape[1]], device=device)
        text_pad = gpt.set_text_padding(text_tokens.clone(), text_len)
        text_pad = F.pad(text_pad, (0, 1), value=gpt.stop_text_token)
        text_in, _ = gpt.build_aligned_inputs_and_targets(
            text_pad, gpt.start_text_token, gpt.stop_text_token
        )
        text_emb = gpt.text_embedding(text_in) + gpt.text_pos_embedding(text_in)

        # Mel embeddings from codes
        mel_pad = F.pad(codes, (0, 1), value=gpt.stop_mel_token)       # [1, T+1]
        mel_in, _ = gpt.build_aligned_inputs_and_targets(
            mel_pad, gpt.start_mel_token, gpt.stop_mel_token
        )  # [1, T+2]
        mel_emb = gpt.mel_embedding(mel_in) + gpt.mel_pos_embedding(mel_in)

        # GPT forward
        emb = torch.cat([conds, text_emb, mel_emb], dim=1)
        gpt_out = gpt.gpt(inputs_embeds=emb, return_dict=True, use_cache=False)
        enc = gpt.final_norm(gpt_out.last_hidden_state[:, conds.shape[1]:])

        # Mel logits at code positions
        mel_enc = enc[:, -mel_emb.shape[1]:]        # [1, T+2, dim]
        mel_logits = gpt.mel_head(mel_enc)           # [1, T+2, vocab]

        log_probs = F.log_softmax(mel_logits, dim=-1)           # [1, T+2, vocab]
        lp = log_probs[:, :T].gather(
            2, codes.unsqueeze(-1)
        ).squeeze(-1)  # [1, T]
        return lp.squeeze(0)  # [T]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_index_tts_on_path() -> None:
    index_tts_path = str(Path(__file__).parent.parent / "index-tts")
    if index_tts_path not in sys.path:
        sys.path.insert(0, index_tts_path)
