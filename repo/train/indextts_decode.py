"""IndexTTS decoder: semantic codes → wav via s2mel + BigVGAN.

Extracts the decode path from infer_v2_modded.py into a reusable function
so the GRPO training loop can score each rollout independently.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchaudio

if TYPE_CHECKING:
    from .indextts_policy import Conditioning, IndexTTSPolicy, RolloutResult


class IndexTTSDecoder:
    def __init__(self, policy: "IndexTTSPolicy") -> None:
        self.policy = policy
        self.tts = policy.tts
        self.device = policy.device

    def decode(
        self,
        rollout: "RolloutResult",
        k: int,
        cond: "Conditioning",
        output_dir: Path | str,
        step: int,
        sample_id: int,
        diffusion_steps: int = 25,
        inference_cfg_rate: float = 0.7,
    ) -> str:
        """Decode rollout.codes[k] to a wav file; return absolute path."""
        tts = self.tts
        device = self.device
        output_dir = Path(output_dir)
        (output_dir / "wav").mkdir(parents=True, exist_ok=True)

        codes = rollout.codes[k].long().unsqueeze(0)                    # [1, T]
        speech_cond_latent = rollout.speech_conditioning_latent          # [1, 32, dim]
        emovec = rollout.emovec                                          # [1, dim]
        text_tokens = rollout.text_tokens.long()                         # [1, L]
        use_speed = rollout.use_speed                                    # [1]

        with torch.no_grad():
            # ── GPT latent forward ──────────────────────────────────────
            latent = tts.gpt(
                speech_cond_latent,
                text_tokens,
                torch.tensor([text_tokens.shape[-1]], device=device),
                codes,
                torch.tensor([codes.shape[-1]], device=device),
                cond.spk_cond_emb,            # emo placeholder (ignored when emo_vec set)
                cond_mel_lengths=torch.tensor([cond.spk_cond_emb.shape[-1]], device=device),
                emo_cond_mel_lengths=torch.tensor([cond.spk_cond_emb.shape[-1]], device=device),
                emo_vec=emovec,
                use_speed=use_speed,
                do_spk_cond=False,
            )  # [1, T_codes, model_dim]

            # ── S2mel decode ─────────────────────────────────────────────
            latent = tts.s2mel.models["gpt_layer"](latent)
            S_infer = tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
            S_infer = S_infer.transpose(1, 2) + latent

            code_len = torch.tensor([codes.shape[-1]], device=device, dtype=torch.long)
            target_lengths = (code_len * 1.72).long()

            cond_mel = tts.s2mel.models["length_regulator"](
                S_infer, ylens=target_lengths, n_quantizers=3, f0=None
            )[0]

            cat_cond = torch.cat([cond.prompt_condition, cond_mel], dim=1)
            mel_lengths = torch.full(
                (cat_cond.size(0),), cat_cond.size(1), dtype=torch.long, device=device
            )

            vc_target = tts.s2mel.models["cfm"].inference(
                cat_cond,
                mel_lengths,
                cond.ref_mel,
                cond.style,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, cond.ref_mel.size(-1):]

            wav = tts.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu()

        out_path = str(output_dir / "wav" / f"step{step:04d}_k{sample_id}.wav")
        torchaudio.save(out_path, wav.type(torch.int16), 22050)
        return out_path
