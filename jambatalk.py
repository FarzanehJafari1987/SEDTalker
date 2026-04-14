# jambatalk.py (updated)
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from wav2vec import Wav2Vec2Model, Wav2Vec2ForCTC, linear_interpolation
from mamba_ssm.modules.mamba_simple import Mamba
from moe_mamba import MambaMoELayer
from transformer import TransformerDecoderLayerGQA_RoPE


class JambaTalk(nn.Module):
    """JambaTalk with Emotion & Intensity Conditioning (robust dtype/shape checks + debug)"""

    def __init__(self, args, debug: bool = False):
        super(JambaTalk, self).__init__()
        self.dataset = args.dataset
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.vertice_dim = args.vertice_dim
        self.debug = debug

        # ---------------- Audio/Text Encoders ----------------
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )
        self.text_encoder = Wav2Vec2ForCTC.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.audio_encoder.feature_extractor._freeze_parameters()

        # ---------------- EmoVOCA Dataset Configs ----------------
        pkl_path = "/media/farzaneh/New Volume/2026_projects/__SEDTalker_2026/EmoVOCA/FLAME_masks.pkl"
        with open(pkl_path, "rb") as f:
            self.lip_mask = pickle.load(f, encoding="latin1")["lips"]

        self.lip_map = nn.Linear(254 * 3, 1024)
        n_q, n_kv = 8, 8
        self.src_fps, self.tgt_fps = 30, 50
        audio_in_size = 1024

        # ---------------- Sequence backbone ----------------
        self.mamba = Mamba(d_model=self.feature_dim)
        self.mamba_moe = MambaMoELayer(
            dim=self.feature_dim, d_state=8, d_conv=8, num_experts=2, num_experts_per_token=2
        )
        self.transformer_decoder = TransformerDecoderLayerGQA_RoPE(
            d_model=self.feature_dim, n_query_heads=n_q, n_kv_heads=n_kv
        )

        # ---------------- Projections ----------------
        self.audio_feature_map = nn.Linear(audio_in_size, self.feature_dim)
        self.vertice_map_r = nn.Linear(self.feature_dim, self.vertice_dim)
        nn.init.constant_(self.vertice_map_r.weight, 0.0)
        nn.init.constant_(self.vertice_map_r.bias, 0.0)

        # ---------------- Lip-text pathway ----------------
        self.transformer = nn.Transformer(d_model=1024, batch_first=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.lm_head = nn.Linear(1024, 33)

        # ---------------- Emotion & Intensity ----------------
        self.num_emotions = getattr(args, 'num_emotions', 6)  # Auto-detect from checkpoint
        self.emotion_emb = nn.Embedding(self.num_emotions, self.feature_dim)
        # intensity is scalar → project to same dimension
        self.intensity_proj = nn.Linear(1, self.feature_dim)

    # ---------- helper stacks ----------
    def _stack_mamba_moe_head(self, x, depth: int):
        for _ in range(depth):
            x = self.mamba(x)
            x = self.mamba_moe(x)
        return x

    def _transformer_block(self, x):
        return self.transformer_decoder(x, x)

    def _stack_mamba_moe_tail(self, x, depth: int):
        for _ in range(depth):
            x = self.mamba_moe(x)
            x = self.mamba(x)
        return x

    def _backbone(self, vertice_input, use_decoder_blocks: int, extra_mamba_blocks: int):
        x = self._stack_mamba_moe_head(vertice_input, depth=use_decoder_blocks)
        x = self._transformer_block(x)
        x = self._stack_mamba_moe_tail(x, depth=extra_mamba_blocks)
        x = self.mamba_moe(x)
        return x

    def _run_sequence_backbone(self, vertice_input):
        # Fixed depth for EmoVOCA
        return self._backbone(vertice_input, use_decoder_blocks=1, extra_mamba_blocks=1)

    # ---------- Conditioning (robust) ----------
    def _condition_features(self, vertice_input, emotion_id, intensity):
        """
        Robust coercion + CPU diagnostics:
        - emotion_id -> long, shape (B,)
        - intensity -> float32, shape (B,1)
        Validate ranges and finiteness on CPU to avoid device-side asserts.
        """
        device = vertice_input.device

        # ---------- coerce emotion_id ----------
        if emotion_id is None:
            raise ValueError("emotion_id is required for conditioning")
        # Move to CPU for validation, but keep a device copy for computation
        emotion_id_dev = emotion_id.to(device=device, dtype=torch.long)
        try:
            emotion_id_cpu = emotion_id_dev.detach().cpu()
        except Exception:
            # ensure we can still make a CPU copy
            emotion_id_cpu = emotion_id.to(device='cpu', dtype=torch.long).detach()

        # Ensure 1D (B,)
        if emotion_id_cpu.ndim > 1 and emotion_id_cpu.size(0) == 1:
            emotion_id_cpu = emotion_id_cpu.view(-1)
            emotion_id_dev = emotion_id_dev.view(-1)

        # ---------- coerce intensity ----------
        if intensity is None:
            raise ValueError("intensity is required for conditioning")
        # Move a CPU copy for validation
        intensity_dev = intensity.to(device=device)
        try:
            intensity_cpu = intensity_dev.detach().cpu().to(dtype=torch.float32)
        except Exception:
            intensity_cpu = intensity.to(device='cpu').detach().to(dtype=torch.float32)

        # Convert shape to (B,1) on device copy
        if intensity_cpu.ndim == 0:
            intensity_cpu = intensity_cpu.view(1, 1)
        elif intensity_cpu.ndim == 1:
            intensity_cpu = intensity_cpu.view(-1, 1)
        elif intensity_cpu.ndim > 2:
            intensity_cpu = intensity_cpu.view(intensity_cpu.size(0), -1)
            if intensity_cpu.size(1) != 1:
                intensity_cpu = intensity_cpu.mean(dim=1, keepdim=True)

        # Validate on CPU to avoid device-side asserts
        # 1) emotion_id ranges
        num_em = self.emotion_emb.num_embeddings
        if (emotion_id_cpu < 0).any() or (emotion_id_cpu >= num_em).any():
            raise ValueError(
                f"emotion_id out of range: min={int(emotion_id_cpu.min())}, "
                f"max={int(emotion_id_cpu.max())}, allowed=[0, {num_em-1}]"
            )

        # 2) intensity finite and reasonable
        if not torch.isfinite(intensity_cpu).all():
            raise ValueError("intensity contains NaN or Inf: %s" % str(intensity_cpu.flatten()[:10].numpy()))
        # Optional: check for huge values
        if torch.abs(intensity_cpu).max() > 1e6:
            raise ValueError("intensity contains very large values: max=%s" % str(torch.abs(intensity_cpu).max().item()))

        # After validating, create device tensors properly shaped/dtyped
        emotion_id_dev = emotion_id_cpu.to(device=device, dtype=torch.long)
        intensity_dev = intensity_cpu.to(device=device, dtype=torch.float32).contiguous()

        # Debug (moves small CPU summaries only)
        if self.debug:
            print(">>> conditioning debug (CPU-validated)")
            print(" emotion_id cpu min/max:", int(emotion_id_cpu.min()), int(emotion_id_cpu.max()))
            print(" intensity cpu shape/vals (head):", intensity_cpu.shape, intensity_cpu.view(-1)[:8].numpy())

        # ---------- compute conditioning ----------
        emo_vec = self.emotion_emb(emotion_id_dev)          # (B, d)
        inten_vec = self.intensity_proj(intensity_dev)      # (B, d)

        cond = emo_vec + inten_vec
        cond = cond.unsqueeze(1).expand(-1, vertice_input.size(1), -1)
        return vertice_input + cond


    # ---------- core forward utilities ----------
    def _prepare_hidden_states(self, audio, frame_num=None):
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        return hidden_states, frame_num

    def _lip_text_path(self, vertice_out, text_hidden_states=None, text_logits=None, out_len_override=None):
        B, T, _ = vertice_out.shape
        lip_pred = vertice_out.view(B, T, -1, 3)[:, :, self.lip_mask, :].reshape(B, T, -1)
        lip_offset = self.lip_map(lip_pred)
        out_len = out_len_override
        lip_offset = linear_interpolation(lip_offset, self.src_fps, 50, output_len=out_len)
        lip_features = self.transformer(lip_offset, lip_offset)
        logits = self.lm_head(self.dropout(lip_features))
        return lip_features, logits

    # ---------- Forward ----------
    def forward(self, audio, template, vertice, emotion_id=None, intensity=None):
        template = template.unsqueeze(1)
        frame_num = vertice.shape[1]

        hidden_states, _ = self._prepare_hidden_states(audio, frame_num=frame_num)
        vertice_input = self.audio_feature_map(hidden_states)

        if (emotion_id is not None) and (intensity is not None):
            vertice_input = self._condition_features(vertice_input, emotion_id, intensity)

        vertice_out = self._run_sequence_backbone(vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        vertice_out = vertice_out + template

        # Text branch
        audio_model = self.text_encoder(audio)
        text_hidden_states = audio_model.hidden_states
        text_logits = audio_model.logits
        out_len = text_hidden_states.shape[1]

        lip_features, logits = self._lip_text_path(
            vertice_out,
            text_hidden_states=text_hidden_states,
            text_logits=text_logits,
            out_len_override=out_len,
        )
        return vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits

    # ---------- Inference ----------
    @torch.no_grad()
    def predict(self, audio, template, emotion_id=None, intensity=None):
        template = template.unsqueeze(1)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        vertice_input = self.audio_feature_map(hidden_states)

        if (emotion_id is not None) and (intensity is not None):
            vertice_input = self._condition_features(vertice_input, emotion_id, intensity)

        vertice_out = self._run_sequence_backbone(vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        vertice_out = vertice_out + template

        lip_features, logits = self._lip_text_path(vertice_out, out_len_override=None)
        return vertice_out, lip_features, logits
