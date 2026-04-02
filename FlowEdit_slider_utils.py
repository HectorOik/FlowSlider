"""
FlowSlider: 3プロンプト方向性分解による連続スケール制御

FlowEditの実画像編集能力を維持しながら、FreeSlidersの考え方を取り入れて
連続的な編集強度制御を可能にする手法。

数式:
    V_steer = V_tar_pos - V_tar_neg  (純粋な編集方向)
    V_fid = V_tar_neg - V_src           (ベース変化)
    V_delta_s = V_fid + strength * V_steer

strength=0: tar_neg方向への編集（例：劣化なし）
strength=1: tar_pos方向への編集（例：完全劣化）
0<strength<1: 連続的な中間状態
"""

from typing import Optional, Tuple, Union, Dict, List, Any
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np
import json
import os

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

# FlowEdit_utils.pyから必要な関数をインポート
from FlowEdit_utils import scale_noise, calculate_shift, calc_v_flux


# ============================================
# Vector Logging and Visualization Utilities
# ============================================

def compute_vector_stats(
    V_fid: torch.Tensor,
    V_steer: torch.Tensor,
    V_delta_s: torch.Tensor,
    zt_edit: torch.Tensor,
    prev_V_steer: Optional[torch.Tensor] = None,
    prev_zt_edit: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute statistics for velocity field vectors at a single timestep.

    Args:
        V_fid: Base velocity (V_neg - V_src)
        V_steer: Direction velocity (V_pos - V_neg)
        V_delta_s: Combined velocity (V_fid + strength * V_steer)
        zt_edit: Current edited latent
        prev_V_steer: V_steer from previous timestep (for cosine similarity)
        prev_zt_edit: zt_edit from previous timestep (for delta computation)

    Returns:
        Dictionary of statistics
    """
    stats = {}

    # Compute norms (average over sequence dimension)
    stats["V_fid_norm"] = V_fid.norm(dim=-1).mean().item()
    stats["V_steer_norm"] = V_steer.norm(dim=-1).mean().item()
    stats["V_delta_s_norm"] = V_delta_s.norm(dim=-1).mean().item()
    stats["zt_edit_norm"] = zt_edit.norm(dim=-1).mean().item()

    # Compute cosine similarity with previous V_steer
    if prev_V_steer is not None:
        # Flatten for cosine similarity computation
        v_dir_flat = V_steer.view(-1)
        prev_v_dir_flat = prev_V_steer.view(-1)
        cos_sim = F.cosine_similarity(v_dir_flat.unsqueeze(0), prev_v_dir_flat.unsqueeze(0)).item()
        stats["V_steer_cosine"] = cos_sim
    else:
        stats["V_steer_cosine"] = 1.0  # First step, no previous

    # Compute angle between V_fid and V_steer
    v_base_flat = V_fid.view(-1)
    v_dir_flat = V_steer.view(-1)
    cos_angle = F.cosine_similarity(v_base_flat.unsqueeze(0), v_dir_flat.unsqueeze(0)).item()
    # Clamp to avoid numerical issues with arccos
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_rad = np.arccos(cos_angle)
    stats["V_fid_V_steer_angle"] = np.degrees(angle_rad)

    # Compute zt_edit delta (movement from previous step)
    if prev_zt_edit is not None:
        delta = (zt_edit - prev_zt_edit).norm(dim=-1).mean().item()
        stats["zt_edit_delta"] = delta
    else:
        stats["zt_edit_delta"] = 0.0

    return stats


def save_vector_stats(
    stats_list: List[Dict[str, Any]],
    output_dir: str,
    strength: float,
):
    """
    Save vector statistics to JSON file.

    Args:
        stats_list: List of statistics dictionaries (one per timestep)
        output_dir: Output directory
        strength: Scale value used
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reorganize data for easier plotting
    output_data = {
        "strength": strength,
        "timesteps": [s["timestep"] for s in stats_list],
        "V_fid_norm": [s["V_fid_norm"] for s in stats_list],
        "V_steer_norm": [s["V_steer_norm"] for s in stats_list],
        "V_delta_s_norm": [s["V_delta_s_norm"] for s in stats_list],
        "V_steer_cosine": [s["V_steer_cosine"] for s in stats_list],
        "V_fid_V_steer_angle": [s["V_fid_V_steer_angle"] for s in stats_list],
        "zt_edit_norm": [s["zt_edit_norm"] for s in stats_list],
        "zt_edit_delta": [s["zt_edit_delta"] for s in stats_list],
    }

    output_path = os.path.join(output_dir, f"stats_scale_{strength:.2f}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return output_path


def plot_vector_stats(
    stats_path: str,
    output_dir: str,
):
    """
    Generate visualization plots from saved statistics.

    Args:
        stats_path: Path to stats JSON file
        output_dir: Output directory for plots
    """
    import matplotlib.pyplot as plt

    with open(stats_path, "r") as f:
        data = json.load(f)

    strength = data["strength"]
    timesteps = data["timesteps"]

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Norms
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, data["V_fid_norm"], 'b-', label="V_fid", linewidth=2)
    ax.plot(timesteps, data["V_steer_norm"], 'r-', label="V_steer", linewidth=2)
    ax.plot(timesteps, data["V_delta_s_norm"], 'g--', label="V_delta_s", linewidth=2)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title(f"Velocity Field Norms (strength={strength:.2f})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # t goes from 1.0 to 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_norms_scale_{strength:.2f}.png"), dpi=150)
    plt.close()

    # Plot 2: V_steer Cosine Similarity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps[1:], data["V_steer_cosine"][1:], 'purple', linewidth=2)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label="Stability threshold (0.9)")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(f"V_steer Directional Consistency (strength={strength:.2f})", fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_cosine_scale_{strength:.2f}.png"), dpi=150)
    plt.close()

    # Plot 3: Angle between V_fid and V_steer
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, data["V_fid_V_steer_angle"], 'orange', linewidth=2)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label="Orthogonal (90°)")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Angle (degrees)", fontsize=12)
    ax.set_title(f"Angle between V_fid and V_steer (strength={strength:.2f})", fontsize=14)
    ax.set_ylim(0, 180)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_angles_scale_{strength:.2f}.png"), dpi=150)
    plt.close()

    # Plot 4: Edit Trajectory (zt_edit movement)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, data["zt_edit_delta"], 'teal', linewidth=2)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Step Movement (L2 norm)", fontsize=12)
    ax.set_title(f"Edit Trajectory: Per-step Movement (strength={strength:.2f})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_trajectory_scale_{strength:.2f}.png"), dpi=150)
    plt.close()


def plot_vector_comparison(
    log_dir: str,
    strengths: List[float],
    output_dir: str,
):
    """
    Generate comparison plots across multiple strengths.

    Args:
        log_dir: Directory containing stats JSON files
        strengths: List of strength values to compare
        output_dir: Output directory for comparison plots
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Load all stats
    all_data = {}
    for strength in strengths:
        stats_path = os.path.join(log_dir, f"stats_scale_{strength:.2f}.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                all_data[strength] = json.load(f)

    if not all_data:
        print(f"No stats files found in {log_dir}")
        return

    # Color map for different strengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(strengths)))

    # Comparison Plot 1: Norms (V_steer)
    fig, ax = plt.subplots(figsize=(12, 7))
    for (strength, data), color in zip(all_data.items(), colors):
        ax.plot(data["timesteps"], data["V_steer_norm"],
                color=color, linewidth=2, label=f"strength={strength:.1f}")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("V_steer L2 Norm", fontsize=12)
    ax.set_title("V_steer Norm Comparison across Scales", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_comparison_norms.png"), dpi=150)
    plt.close()

    # Comparison Plot 2: Cosine Similarity
    fig, ax = plt.subplots(figsize=(12, 7))
    for (strength, data), color in zip(all_data.items(), colors):
        ax.plot(data["timesteps"][1:], data["V_steer_cosine"][1:],
                color=color, linewidth=2, label=f"strength={strength:.1f}")
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("V_steer Directional Consistency Comparison", fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_comparison_cosine.png"), dpi=150)
    plt.close()

    # Comparison Plot 3: Angles
    fig, ax = plt.subplots(figsize=(12, 7))
    for (strength, data), color in zip(all_data.items(), colors):
        ax.plot(data["timesteps"], data["V_fid_V_steer_angle"],
                color=color, linewidth=2, label=f"strength={strength:.1f}")
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Angle (degrees)", fontsize=12)
    ax.set_title("V_fid-V_steer Angle Comparison", fontsize=14)
    ax.set_ylim(0, 180)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_comparison_angles.png"), dpi=150)
    plt.close()

    # Comparison Plot 4: Trajectory (cumulative movement)
    fig, ax = plt.subplots(figsize=(12, 7))
    for (strength, data), color in zip(all_data.items(), colors):
        cumulative = np.cumsum(data["zt_edit_delta"])
        ax.plot(data["timesteps"], cumulative,
                color=color, linewidth=2, label=f"strength={strength:.1f}")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Cumulative Movement", fontsize=12)
    ax.set_title("Edit Trajectory: Cumulative Distance from Source", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_comparison_trajectory.png"), dpi=150)
    plt.close()

    print(f"Comparison plots saved to {output_dir}")


def prepare_mask_for_flux(
    mask: torch.Tensor,
    target_height: int,
    target_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Prepare a binary mask for use with Flux's packed latent format.

    Args:
        mask: Input mask tensor. Can be:
            - (H, W): Single channel 2D mask
            - (1, H, W): Single channel with batch dim
            - (B, 1, H, W): Full 4D tensor
            - (B, H, W): 3D tensor
        target_height: Target height in latent space (H/8)
        target_width: Target width in latent space (W/8)
        device: Target device
        dtype: Target dtype

    Returns:
        Mask tensor of shape (1, seq_len, 1) for packed latent format
        where seq_len = (H/2) * (W/2) for Flux's 2x2 packing
    """
    # Ensure 4D tensor (B, C, H, W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    elif mask.dim() == 3:
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(0)  # (1, H, W) -> (1, 1, H, W)
        else:
            mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

    mask = mask.to(device=device, dtype=dtype)

    # Resize to latent space dimensions
    mask_resized = F.interpolate(
        mask,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )

    # For Flux: pack into sequence format
    # Flux uses 2x2 packing, so (B, C, H, W) -> (B, H/2 * W/2, C*4)
    # For mask, we just need (B, seq_len, 1) where values are averaged over 2x2 patches
    B, C, H, W = mask_resized.shape

    # Reshape for 2x2 packing: (B, 1, H, W) -> (B, H//2, 2, W//2, 2) -> (B, H//2, W//2, 4)
    mask_packed = mask_resized.view(B, C, H // 2, 2, W // 2, 2)
    mask_packed = mask_packed.permute(0, 2, 4, 1, 3, 5)  # (B, H//2, W//2, C, 2, 2)
    mask_packed = mask_packed.reshape(B, (H // 2) * (W // 2), C * 4)  # (B, seq_len, 4)

    # Average across the 4 values in each 2x2 patch to get single mask value
    mask_packed = mask_packed.mean(dim=-1, keepdim=True)  # (B, seq_len, 1)

    return mask_packed


@torch.no_grad()
def FlowEditFLUX_Slider(
    pipe,
    scheduler,
    x_src,
    src_prompt: str,
    tar_prompt: str,
    tar_prompt_neg: str,
    negative_prompt: str = "",
    strength: float = 1.0,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
    scale_mode: str = "slider",
    normalize_v_dir: bool = False,
    v_dir_target_norm: float = 1.0,
    log_vectors: bool = False,
    log_output_dir: Optional[str] = None,
):
    """
    FlowEdit with 3-prompt directional decomposition for continuous strength control.

    Args:
        pipe: FluxPipeline
        scheduler: FlowMatchEulerDiscreteScheduler
        x_src: Source image latent (B, C, H, W)
        src_prompt: Source prompt describing the original image (e.g., "a building")
        tar_prompt: Positive target prompt (e.g., "a severely decayed building")
        tar_prompt_neg: Negative target prompt (e.g., "a new building")
        negative_prompt: Negative prompt for CFG (usually empty for Flux)
        strength: Edit intensity strength (0.0 = tar_neg direction, 1.0 = tar_pos direction)
        T_steps: Total number of timesteps
        n_avg: Number of velocity field averaging iterations
        src_guidance_scale: Guidance strength for source prompt
        tar_guidance_scale: Guidance strength for target prompts
        n_min: Number of final steps using regular sampling
        n_max: Maximum number of steps to apply flow editing
        scale_mode: Scaling method - "slider" (default), "interp", "step", "cfg", or "direct"
            - "slider": Scale the direction vector V_delta_s = V_fid + strength * V_steer (FreeSlider-like)
            - "interp": FlowEdit-based interpolation V_final = V_src + strength * (V_pos - V_src)
            - "step": Scale the step size dt (FlowEdit paper experiment, causes degradation)
            - "cfg": Scale the target guidance (tar_guidance_scale * strength)
            - "direct": Scale the full velocity difference V_delta_s = strength * (V_pos - V_src) without decomposition
        normalize_v_dir: If True, normalize V_steer to v_dir_target_norm before scaling.
            This stabilizes edit strength across different CFG settings and prevents
            both numerical instability (low CFG) and semantic over-editing (high CFG).
            Only applies when scale_mode="slider".
        v_dir_target_norm: Target L2 norm for V_steer normalization (default: 1.0).
            Higher values produce stronger edits per unit strength.
        log_vectors: If True, record vector statistics and generate visualization plots
        log_output_dir: Output directory for vector logs (required if log_vectors=True)

    Returns:
        Edited latent tensor
    """
    # Validate log_vectors arguments
    if log_vectors and log_output_dir is None:
        raise ValueError("log_output_dir must be specified when log_vectors=True")

    # Initialize logging variables
    stats_list = [] if log_vectors else None
    prev_V_steer = None
    prev_zt_edit = None

    device = x_src.device
    # Note: orig_height/width should match the actual image dimensions for correct latent_image_ids
    # x_src is VAE-encoded latent (H/8, W/8), so multiply by vae_scale_factor to get original size
    orig_height = x_src.shape[2] * pipe.vae_scale_factor
    orig_width = x_src.shape[3] * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    # Prepare latents
    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src.dtype,
        device=x_src.device,
        generator=None,
        latents=x_src
    )
    x_src_packed = pipe._pack_latents(
        x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3]
    )
    latent_image_ids = latent_src_image_ids

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )

    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # ============================================
    # Encode prompts (3 prompts)
    # ============================================

    # Source prompt
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # Target positive prompt (e.g., "severely decayed")
    (
        tar_pos_prompt_embeds,
        tar_pos_pooled_prompt_embeds,
        tar_pos_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # Target negative prompt (e.g., "new, pristine")
    (
        tar_neg_prompt_embeds,
        tar_neg_pooled_prompt_embeds,
        tar_neg_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt_neg,
        prompt_2=None,
        device=device,
    )

    # ============================================
    # Handle guidance
    # ============================================
    # For cfg mode, strength is applied to tar_guidance_scale
    effective_tar_guidance = tar_guidance_scale * strength if scale_mode == "cfg" else tar_guidance_scale

    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_pos_guidance = torch.tensor([effective_tar_guidance], device=device)
        tar_pos_guidance = tar_pos_guidance.expand(x_src_packed.shape[0])
        tar_neg_guidance = torch.tensor([effective_tar_guidance], device=device)
        tar_neg_guidance = tar_neg_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_pos_guidance = None
        tar_neg_guidance = None

    # Initialize ODE: zt_edit = x_src
    zt_edit = x_src_packed.clone()

    # ============================================
    # Main editing loop
    # ============================================
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc=f"FlowSlider (strength={strength:.2f})"):

        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        if T_steps - i > n_min:
            # Flow-based editing phase

            V_delta_s_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):
                # Forward noise
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)

                # Source trajectory
                zt_src = (1 - t_i) * x_src_packed + t_i * fwd_noise

                # Target trajectory (with offset preservation)
                zt_tar = zt_edit + zt_src - x_src_packed

                # ============================================
                # 3-prompt velocity computation (CORE CHANGE)
                # ============================================

                # Source velocity
                Vt_src = calc_v_flux(
                    pipe,
                    latents=zt_src,
                    prompt_embeds=src_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t
                )

                # Positive target velocity
                Vt_pos = calc_v_flux(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_pos_prompt_embeds,
                    pooled_prompt_embeds=tar_pos_pooled_prompt_embeds,
                    guidance=tar_pos_guidance,
                    text_ids=tar_pos_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t
                )

                # Negative target velocity
                Vt_neg = calc_v_flux(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_neg_prompt_embeds,
                    pooled_prompt_embeds=tar_neg_pooled_prompt_embeds,
                    guidance=tar_neg_guidance,
                    text_ids=tar_neg_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t
                )

                # ============================================
                # Directional decomposition
                # ============================================
                # V_steer: Pure edit direction (e.g., "aging" direction)
                V_steer = Vt_pos - Vt_neg

                # V_fid: Base change from source to negative target
                V_fid = Vt_neg - Vt_src

                # V_delta_s computation depends on scale_mode
                if scale_mode == "slider":
                    # Slider mode: strength the direction vector (FreeSlider-like)
                    # strength=0 -> V_fid only (tar_neg direction)
                    # strength=1 -> V_fid + V_steer = Vt_pos - Vt_src (tar_pos direction)

                    # Apply V_steer normalization if enabled
                    if normalize_v_dir:
                        # Compute current norm (mean over sequence dimension)
                        v_dir_norm = V_steer.norm(dim=-1, keepdim=True).mean()
                        # Normalize to target norm
                        V_steer_scaled = V_steer * (v_dir_target_norm / (v_dir_norm + 1e-8))
                    else:
                        V_steer_scaled = V_steer

                    V_delta_s = V_fid + strength * V_steer_scaled
                elif scale_mode == "direct":
                    # Direct mode: strength the full velocity difference without decomposition
                    # V_delta_s = strength * (V_pos - V_src)
                    # This strengths both V_fid and V_steer together, causing trajectory collapse at strength > 1
                    V_delta_full = Vt_pos - Vt_src
                    V_delta_s = strength * V_delta_full
                elif scale_mode == "interp":
                    # Interp mode: FlowEdit-based interpolation
                    # V_final = V_src + strength * (V_pos - V_src) = (1-strength)*V_src + strength*V_pos
                    # For 3-prompt: V_final = V_src + strength * (V_pos - V_neg) + (V_neg - V_src)
                    #                       = V_neg + strength * V_steer
                    # But we want: V_final = V_src + strength * V_delta_full
                    # where V_delta_full = V_pos - V_src (full edit direction)
                    V_delta_full = Vt_pos - Vt_src
                    V_final = Vt_src + strength * V_delta_full
                    # Store V_final directly, will be used differently in ODE propagation
                    V_delta_s = V_final  # This is actually V_final, not a delta
                elif scale_mode == "step":
                    # Step mode: V_delta_s is fixed at strength=1, dt will be scaled later
                    V_delta_s = V_fid + V_steer  # equivalent to strength=1
                elif scale_mode == "cfg":
                    # CFG mode: strength was already applied to guidance, use strength=1 for direction
                    V_delta_s = V_fid + V_steer  # equivalent to strength=1
                else:
                    raise ValueError(f"Unknown scale_mode: {scale_mode}")

                V_delta_s_avg += (1 / n_avg) * V_delta_s

            # ============================================
            # Vector Logging (if enabled)
            # ============================================
            if log_vectors:
                # Use the last computed V_fid and V_steer for logging
                # (when n_avg > 1, this is from the last iteration)
                # Note: For slider mode with normalize_v_dir, log the original V_steer
                # to see the raw values before normalization
                step_stats = compute_vector_stats(
                    V_fid=V_fid,
                    V_steer=V_steer,
                    V_delta_s=V_delta_s_avg,
                    zt_edit=zt_edit,
                    prev_V_steer=prev_V_steer,
                    prev_zt_edit=prev_zt_edit,
                )
                step_stats["timestep"] = t_i.item() if hasattr(t_i, 'item') else float(t_i)
                # Log normalization info if enabled
                if normalize_v_dir and scale_mode == "slider":
                    step_stats["normalize_v_dir"] = True
                    step_stats["v_dir_target_norm"] = v_dir_target_norm
                    step_stats["v_dir_original_norm"] = V_steer.norm(dim=-1).mean().item()
                stats_list.append(step_stats)

                # Store current values for next iteration comparison
                prev_V_steer = V_steer.clone()
                prev_zt_edit = zt_edit.clone()

            # Propagate ODE
            zt_edit = zt_edit.to(torch.float32)
            if scale_mode == "step":
                # Step mode: strength the step size dt (FlowEdit paper experiment)
                zt_edit = zt_edit + strength * (t_im1 - t_i) * V_delta_s_avg
            elif scale_mode == "interp":
                # Interp mode: V_delta_s_avg is actually V_final, use directly
                zt_edit = zt_edit + (t_im1 - t_i) * V_delta_s_avg
            else:
                # Slider and CFG mode: normal dt
                zt_edit = zt_edit + (t_im1 - t_i) * V_delta_s_avg
            zt_edit = zt_edit.to(V_delta_s_avg.dtype)

        else:  # Regular sampling for last n_min steps
            if i == T_steps - n_min:
                # Initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            # For final steps, use interpolated target based on strength
            # Interpolate between neg and pos embeddings
            interp_prompt_embeds = (1 - strength) * tar_neg_prompt_embeds + strength * tar_pos_prompt_embeds
            interp_pooled_embeds = (1 - strength) * tar_neg_pooled_prompt_embeds + strength * tar_pos_pooled_prompt_embeds

            Vt_tar = calc_v_flux(
                pipe,
                latents=xt_tar,
                prompt_embeds=interp_prompt_embeds,
                pooled_prompt_embeds=interp_pooled_embeds,
                guidance=tar_pos_guidance,
                text_ids=tar_pos_text_ids,  # text_ids are typically the same
                latent_image_ids=latent_image_ids,
                t=t
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)

    # ============================================
    # Save and visualize vector statistics
    # ============================================
    if log_vectors and stats_list:
        stats_path = save_vector_stats(stats_list, log_output_dir, strength)
        plot_vector_stats(stats_path, log_output_dir)
        print(f"Vector statistics saved to {log_output_dir}")

    return unpacked_out


@torch.no_grad()
def FlowEditFLUX_Slider_batch(
    pipe,
    scheduler,
    x_src,
    src_prompt: str,
    tar_prompt: str,
    tar_prompt_neg: str,
    negative_prompt: str = "",
    strengths: list = [0.0, 0.25, 0.5, 0.75, 1.0],
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
):
    """
    Batch processing for multiple strengths.
    More efficient than calling FlowEditFLUX_Slider multiple times
    as prompt encoding is done only once.

    Args:
        strengths: List of strength values to generate
        (other args same as FlowEditFLUX_Slider)

    Returns:
        Dict[float, Tensor]: Mapping from strength to edited latent
    """
    results = {}

    for strength in strengths:
        result = FlowEditFLUX_Slider(
            pipe=pipe,
            scheduler=scheduler,
            x_src=x_src,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            tar_prompt_neg=tar_prompt_neg,
            negative_prompt=negative_prompt,
            strength=strength,
            T_steps=T_steps,
            n_avg=n_avg,
            src_guidance_scale=src_guidance_scale,
            tar_guidance_scale=tar_guidance_scale,
            n_min=n_min,
            n_max=n_max,
        )
        results[strength] = result

    return results


@torch.no_grad()
def FlowEditFLUX_Slider_2prompt(
    pipe,
    scheduler,
    x_src,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str = "",
    strength: float = 1.0,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
):
    """
    FlowEdit with 2-prompt simple scaling (without neg prompt).

    数式: V_delta_s = strength * (V_tar - V_src)

    strength=0: 編集なし（元画像のまま）
    strength=1: 通常のFlowEdit（tar方向への完全編集）
    strength>1: tar方向への過剰編集

    Args:
        pipe: FluxPipeline
        scheduler: FlowMatchEulerDiscreteScheduler
        x_src: Source image latent
        src_prompt: Source prompt
        tar_prompt: Target prompt (e.g., "a decayed building")
        strength: Edit intensity (0=no change, 1=full edit)
    """
    device = x_src.device
    orig_height = x_src.shape[2] * pipe.vae_scale_factor
    orig_width = x_src.shape[3] * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    # Prepare latents
    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src.dtype,
        device=x_src.device,
        generator=None,
        latents=x_src
    )
    x_src_packed = pipe._pack_latents(
        x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3]
    )
    latent_image_ids = latent_src_image_ids

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )

    # Encode prompts (2 prompts only)
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # Handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # Initialize ODE
    zt_edit = x_src_packed.clone()

    # Main editing loop
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc=f"FlowEdit-2prompt (strength={strength:.2f})"):

        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        if T_steps - i > n_min:
            V_delta_s_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                zt_src = (1 - t_i) * x_src_packed + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src_packed

                # 2-prompt velocity computation
                Vt_src = calc_v_flux(
                    pipe,
                    latents=zt_src,
                    prompt_embeds=src_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t
                )

                Vt_tar = calc_v_flux(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t
                )

                # Simple scaling: V_delta_s = strength * (V_tar - V_src)
                V_delta_s = strength * (Vt_tar - Vt_src)
                V_delta_s_avg += (1 / n_avg) * V_delta_s

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_s_avg
            zt_edit = zt_edit.to(V_delta_s_avg.dtype)

        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            # Prompt interpolation for stability at high strengths
            # interp = (1 - strength) * src + strength * tar
            interp_prompt_embeds = (1 - strength) * src_prompt_embeds + strength * tar_prompt_embeds
            interp_pooled_embeds = (1 - strength) * src_pooled_prompt_embeds + strength * tar_pooled_prompt_embeds

            Vt_tar = calc_v_flux(
                pipe,
                latents=xt_tar,
                prompt_embeds=interp_prompt_embeds,
                pooled_prompt_embeds=interp_pooled_embeds,
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids=latent_image_ids,
                t=t
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)

    return unpacked_out


@torch.no_grad()
def FlowEditFLUX_Slider_with_mask(
    pipe,
    scheduler,
    x_src,
    src_prompt: str,
    tar_prompt: str,
    tar_prompt_neg: str,
    mask: torch.Tensor,
    negative_prompt: str = "",
    strength: float = 1.0,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
    scale_mode: str = "slider",
):
    """
    FlowEdit with 3-prompt directional decomposition and mask-based local editing.

    This function applies edits only to the masked region, preserving the
    unmasked areas from the source image.

    Args:
        pipe: FluxPipeline
        scheduler: FlowMatchEulerDiscreteScheduler
        x_src: Source image latent (B, C, H, W)
        src_prompt: Source prompt describing the original image
        tar_prompt: Positive target prompt (e.g., "a severely decayed building")
        tar_prompt_neg: Negative target prompt (e.g., "a new building")
        mask: Binary mask tensor indicating edit region.
              Shape: (H, W), (1, H, W), (B, H, W), or (B, 1, H, W)
              Values: 1 = edit region, 0 = preserve original
        negative_prompt: Negative prompt for CFG (usually empty for Flux)
        strength: Edit intensity strength (0.0 = tar_neg direction, 1.0 = tar_pos direction)
        T_steps: Total number of timesteps
        n_avg: Number of velocity field averaging iterations
        src_guidance_scale: Guidance strength for source prompt
        tar_guidance_scale: Guidance strength for target prompts
        n_min: Number of final steps using regular sampling
        n_max: Maximum number of steps to apply flow editing
        scale_mode: Scaling method - "slider" (default), "interp", "step", or "cfg"

    Returns:
        Edited latent tensor with edits applied only in masked region
    """
    device = x_src.device
    orig_height = x_src.shape[2] * pipe.vae_scale_factor
    orig_width = x_src.shape[3] * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    # Prepare latents
    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src.dtype,
        device=x_src.device,
        generator=None,
        latents=x_src
    )
    x_src_packed = pipe._pack_latents(
        x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3]
    )
    latent_image_ids = latent_src_image_ids

    # Prepare mask for packed latent format
    mask_packed = prepare_mask_for_flux(
        mask=mask,
        target_height=x_src.shape[2],
        target_width=x_src.shape[3],
        device=device,
        dtype=x_src.dtype,
    )

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )

    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # Encode prompts (3 prompts)
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    (
        tar_pos_prompt_embeds,
        tar_pos_pooled_prompt_embeds,
        tar_pos_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    (
        tar_neg_prompt_embeds,
        tar_neg_pooled_prompt_embeds,
        tar_neg_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt_neg,
        prompt_2=None,
        device=device,
    )

    # Handle guidance
    # For cfg mode, strength is applied to tar_guidance_scale
    effective_tar_guidance = tar_guidance_scale * strength if scale_mode == "cfg" else tar_guidance_scale

    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_pos_guidance = torch.tensor([effective_tar_guidance], device=device)
        tar_pos_guidance = tar_pos_guidance.expand(x_src_packed.shape[0])
        tar_neg_guidance = torch.tensor([effective_tar_guidance], device=device)
        tar_neg_guidance = tar_neg_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_pos_guidance = None
        tar_neg_guidance = None

    # Initialize ODE: zt_edit = x_src
    zt_edit = x_src_packed.clone()

    # Main editing loop
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc=f"FlowEdit-Slider-Mask (strength={strength:.2f})"):

        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        if T_steps - i > n_min:
            # Flow-based editing phase

            V_delta_s_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):
                # Forward noise
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)

                # Source trajectory
                zt_src = (1 - t_i) * x_src_packed + t_i * fwd_noise

                # Target trajectory (with offset preservation)
                zt_tar = zt_edit + zt_src - x_src_packed

                # 3-prompt velocity computation
                Vt_src = calc_v_flux(
                    pipe,
                    latents=zt_src,
                    prompt_embeds=src_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t
                )

                Vt_pos = calc_v_flux(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_pos_prompt_embeds,
                    pooled_prompt_embeds=tar_pos_pooled_prompt_embeds,
                    guidance=tar_pos_guidance,
                    text_ids=tar_pos_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t
                )

                Vt_neg = calc_v_flux(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_neg_prompt_embeds,
                    pooled_prompt_embeds=tar_neg_pooled_prompt_embeds,
                    guidance=tar_neg_guidance,
                    text_ids=tar_neg_text_ids,
                    latent_image_ids=latent_image_ids,
                    t=t
                )

                # Directional decomposition
                V_steer = Vt_pos - Vt_neg
                V_fid = Vt_neg - Vt_src

                # V_delta_s computation depends on scale_mode
                if scale_mode == "slider":
                    V_delta_s = V_fid + strength * V_steer
                elif scale_mode == "interp":
                    # Interp mode: FlowEdit-based interpolation
                    V_delta_full = Vt_pos - Vt_src
                    V_final = Vt_src + strength * V_delta_full
                    V_delta_s = V_final  # This is actually V_final, not a delta
                elif scale_mode == "step":
                    V_delta_s = V_fid + V_steer  # equivalent to strength=1
                elif scale_mode == "cfg":
                    V_delta_s = V_fid + V_steer  # equivalent to strength=1
                else:
                    raise ValueError(f"Unknown scale_mode: {scale_mode}")

                V_delta_s_avg += (1 / n_avg) * V_delta_s

            # Propagate ODE (without mask first)
            zt_edit = zt_edit.to(torch.float32)
            if scale_mode == "step":
                # Step mode: strength the step size dt
                zt_edit_new = zt_edit + strength * (t_im1 - t_i) * V_delta_s_avg
            elif scale_mode == "interp":
                # Interp mode: V_delta_s_avg is actually V_final, use directly
                zt_edit_new = zt_edit + (t_im1 - t_i) * V_delta_s_avg
            else:
                zt_edit_new = zt_edit + (t_im1 - t_i) * V_delta_s_avg

            # ============================================
            # MASK APPLICATION (LocalBlend style):
            # Apply mask to the RESULT, not the velocity
            # zt_edit = x_src + mask * (zt_edit_new - x_src)
            # This forces unmasked regions to stay at source
            # ============================================
            zt_edit = x_src_packed + mask_packed * (zt_edit_new - x_src_packed)
            zt_edit = zt_edit.to(V_delta_s_avg.dtype)

        else:  # Regular sampling for last n_min steps
            if i == T_steps - n_min:
                # Initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            # Interpolate between neg and pos embeddings
            interp_prompt_embeds = (1 - strength) * tar_neg_prompt_embeds + strength * tar_pos_prompt_embeds
            interp_pooled_embeds = (1 - strength) * tar_neg_pooled_prompt_embeds + strength * tar_pos_pooled_prompt_embeds

            Vt_tar = calc_v_flux(
                pipe,
                latents=xt_tar,
                prompt_embeds=interp_prompt_embeds,
                pooled_prompt_embeds=interp_pooled_embeds,
                guidance=tar_pos_guidance,
                text_ids=tar_pos_text_ids,
                latent_image_ids=latent_image_ids,
                t=t
            )

            xt_tar = xt_tar.to(torch.float32)
            xt_tar_new = xt_tar + (t_im1 - t_i) * Vt_tar

            # LocalBlend style mask application for n_min phase
            xt_tar = x_src_packed + mask_packed * (xt_tar_new - x_src_packed)
            xt_tar = xt_tar.to(Vt_tar.dtype)

    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)

    return unpacked_out


# ============================================
# SD3 Slider Implementation
# ============================================

@torch.no_grad()
def FlowEditSD3_Slider(
    pipe,
    scheduler,
    x_src,
    src_prompt: str,
    tar_prompt: str,
    tar_prompt_neg: str,
    negative_prompt: str = "",
    strength: float = 1.0,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 33,
    scale_mode: str = "slider",
    normalize_v_dir: bool = False,
    v_dir_target_norm: float = 1.0,
    log_vectors: bool = False,
    log_output_dir: Optional[str] = None,
):
    """
    FlowSlider for SD3 with 3-prompt directional decomposition.

    Uses 6-way CFG batching for efficient computation:
    - [src_uncond, src_cond, tar_pos_uncond, tar_pos_cond, tar_neg_uncond, tar_neg_cond]

    Args:
        pipe: StableDiffusion3Pipeline
        scheduler: Scheduler (typically FlowMatchEulerDiscreteScheduler)
        x_src: Source image latent (B, C, H, W)
        src_prompt: Source prompt describing the original image
        tar_prompt: Positive target prompt (e.g., "a severely decayed building")
        tar_prompt_neg: Negative target prompt (e.g., "a new building")
        negative_prompt: Negative prompt for CFG (usually empty)
        strength: Edit intensity strength (0.0 = tar_neg direction, 1.0 = tar_pos direction)
        T_steps: Total number of timesteps (default: 50 for SD3)
        n_avg: Number of velocity field averaging iterations
        src_guidance_scale: Guidance strength for source prompt (default: 3.5 for SD3)
        tar_guidance_scale: Guidance strength for target prompts (default: 13.5 for SD3)
        n_min: Number of final steps using regular sampling
        n_max: Maximum number of steps to apply flow editing (default: 33 for SD3)
        scale_mode: Scaling method - "slider" (default), "interp", "step", "cfg", or "direct"
            - "slider": Scale the direction vector V_delta_s = V_fid + strength * V_steer
            - "direct": Scale the full velocity difference V_delta_s = strength * (V_pos - V_src) without decomposition
        normalize_v_dir: If True, normalize V_steer to v_dir_target_norm before scaling
        v_dir_target_norm: Target L2 norm for V_steer normalization
        log_vectors: If True, record vector statistics
        log_output_dir: Output directory for vector logs

    Returns:
        Edited latent tensor
    """
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    # Validate log_vectors arguments
    if log_vectors and log_output_dir is None:
        raise ValueError("log_output_dir must be specified when log_vectors=True")

    # Initialize logging variables
    stats_list = [] if log_vectors else None
    prev_V_steer = None
    prev_zt_edit = None

    device = x_src.device

    # Retrieve timesteps
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # ============================================
    # Encode prompts (3 prompts with CFG)
    # ============================================

    # Source prompt
    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # Target positive prompt
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_pos_prompt_embeds,
        tar_pos_negative_prompt_embeds,
        tar_pos_pooled_prompt_embeds,
        tar_pos_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # Target negative prompt
    (
        tar_neg_prompt_embeds,
        tar_neg_negative_prompt_embeds,
        tar_neg_pooled_prompt_embeds,
        tar_neg_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt_neg,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # ============================================
    # Prepare 6-way CFG embeddings
    # [src_uncond, src_cond, tar_pos_uncond, tar_pos_cond, tar_neg_uncond, tar_neg_cond]
    # ============================================
    all_prompt_embeds = torch.cat([
        src_negative_prompt_embeds,      # src_uncond
        src_prompt_embeds,               # src_cond
        tar_pos_negative_prompt_embeds,  # tar_pos_uncond
        tar_pos_prompt_embeds,           # tar_pos_cond
        tar_neg_negative_prompt_embeds,  # tar_neg_uncond
        tar_neg_prompt_embeds,           # tar_neg_cond
    ], dim=0)

    all_pooled_prompt_embeds = torch.cat([
        src_negative_pooled_prompt_embeds,
        src_pooled_prompt_embeds,
        tar_pos_negative_pooled_prompt_embeds,
        tar_pos_pooled_prompt_embeds,
        tar_neg_negative_pooled_prompt_embeds,
        tar_neg_pooled_prompt_embeds,
    ], dim=0)

    # Initialize ODE: zt_edit = x_src
    zt_edit = x_src.clone()

    # ============================================
    # Main editing loop
    # ============================================
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc=f"SD3-FlowSlider (strength={strength:.2f})"):

        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:
            # Flow-based editing phase

            V_delta_s_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                # Forward noise
                fwd_noise = torch.randn_like(x_src).to(x_src.device)

                # Source trajectory
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise

                # Target trajectory (with offset preservation)
                zt_tar = zt_edit + zt_src - x_src

                # ============================================
                # 6-way CFG batched computation
                # ============================================
                # Latents: [zt_src, zt_src, zt_tar, zt_tar, zt_tar, zt_tar]
                all_latents = torch.cat([
                    zt_src, zt_src,  # src_uncond, src_cond
                    zt_tar, zt_tar,  # tar_pos_uncond, tar_pos_cond
                    zt_tar, zt_tar,  # tar_neg_uncond, tar_neg_cond
                ])

                # Timestep broadcast
                timestep_batch = t.expand(all_latents.shape[0])

                # Single transformer call
                with torch.no_grad():
                    noise_pred_all = pipe.transformer(
                        hidden_states=all_latents,
                        timestep=timestep_batch,
                        encoder_hidden_states=all_prompt_embeds,
                        pooled_projections=all_pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                # Split into 6 parts
                (
                    src_noise_uncond, src_noise_cond,
                    tar_pos_noise_uncond, tar_pos_noise_cond,
                    tar_neg_noise_uncond, tar_neg_noise_cond,
                ) = noise_pred_all.chunk(6)

                # Apply CFG to get 3 velocities
                Vt_src = src_noise_uncond + src_guidance_scale * (src_noise_cond - src_noise_uncond)
                Vt_pos = tar_pos_noise_uncond + tar_guidance_scale * (tar_pos_noise_cond - tar_pos_noise_uncond)
                Vt_neg = tar_neg_noise_uncond + tar_guidance_scale * (tar_neg_noise_cond - tar_neg_noise_uncond)

                # ============================================
                # Directional decomposition (same as FLUX)
                # ============================================
                V_steer = Vt_pos - Vt_neg
                V_fid = Vt_neg - Vt_src

                # V_delta_s computation depends on scale_mode
                if scale_mode == "slider":
                    if normalize_v_dir:
                        v_dir_norm = V_steer.norm(dim=-1, keepdim=True).mean()
                        V_steer_scaled = V_steer * (v_dir_target_norm / (v_dir_norm + 1e-8))
                    else:
                        V_steer_scaled = V_steer
                    V_delta_s = V_fid + strength * V_steer_scaled
                elif scale_mode == "direct":
                    # Direct mode: strength the full velocity difference without decomposition
                    V_delta_full = Vt_pos - Vt_src
                    V_delta_s = strength * V_delta_full
                elif scale_mode == "interp":
                    V_delta_full = Vt_pos - Vt_src
                    V_final = Vt_src + strength * V_delta_full
                    V_delta_s = V_final
                elif scale_mode == "step":
                    V_delta_s = V_fid + V_steer
                elif scale_mode == "cfg":
                    V_delta_s = V_fid + V_steer
                else:
                    raise ValueError(f"Unknown scale_mode: {scale_mode}")

                V_delta_s_avg += (1 / n_avg) * V_delta_s

            # ============================================
            # Vector Logging (if enabled)
            # ============================================
            if log_vectors:
                step_stats = compute_vector_stats(
                    V_fid=V_fid,
                    V_steer=V_steer,
                    V_delta_s=V_delta_s_avg,
                    zt_edit=zt_edit,
                    prev_V_steer=prev_V_steer,
                    prev_zt_edit=prev_zt_edit,
                )
                step_stats["timestep"] = t_i.item() if hasattr(t_i, 'item') else float(t_i)
                if normalize_v_dir and scale_mode == "slider":
                    step_stats["normalize_v_dir"] = True
                    step_stats["v_dir_target_norm"] = v_dir_target_norm
                    step_stats["v_dir_original_norm"] = V_steer.norm(dim=-1).mean().item()
                stats_list.append(step_stats)
                prev_V_steer = V_steer.clone()
                prev_zt_edit = zt_edit.clone()

            # Propagate ODE
            zt_edit = zt_edit.to(torch.float32)
            if scale_mode == "step":
                zt_edit = zt_edit + strength * (t_im1 - t_i) * V_delta_s_avg
            else:
                zt_edit = zt_edit + (t_im1 - t_i) * V_delta_s_avg
            zt_edit = zt_edit.to(V_delta_s_avg.dtype)

        else:  # Regular sampling for last n_min steps
            if i == T_steps - n_min:
                # Initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            # For final steps, use interpolated target
            interp_prompt_embeds = (1 - strength) * tar_neg_prompt_embeds + strength * tar_pos_prompt_embeds
            interp_pooled_embeds = (1 - strength) * tar_neg_pooled_prompt_embeds + strength * tar_pos_pooled_prompt_embeds

            # 2-way CFG for interpolated target
            interp_all_embeds = torch.cat([tar_pos_negative_prompt_embeds, interp_prompt_embeds], dim=0)
            interp_all_pooled = torch.cat([tar_pos_negative_pooled_prompt_embeds, interp_pooled_embeds], dim=0)
            interp_latents = torch.cat([xt_tar, xt_tar])

            timestep_batch = t.expand(2)

            with torch.no_grad():
                noise_pred_interp = pipe.transformer(
                    hidden_states=interp_latents,
                    timestep=timestep_batch,
                    encoder_hidden_states=interp_all_embeds,
                    pooled_projections=interp_all_pooled,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            interp_uncond, interp_cond = noise_pred_interp.chunk(2)
            Vt_tar = interp_uncond + tar_guidance_scale * (interp_cond - interp_uncond)

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    out = zt_edit if n_min == 0 else xt_tar

    # ============================================
    # Save and visualize vector statistics
    # ============================================
    if log_vectors and stats_list:
        stats_path = save_vector_stats(stats_list, log_output_dir, strength)
        plot_vector_stats(stats_path, log_output_dir)
        print(f"Vector statistics saved to {log_output_dir}")

    return out


# ============================================
# Z-Image Slider Implementation
# ============================================

@torch.no_grad()
def FlowEditZImage_Slider(
    pipe,
    scheduler,
    x_src,
    src_prompt: str,
    tar_prompt: str,
    tar_prompt_neg: str,
    negative_prompt: str = "",
    strength: float = 1.0,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 2.0,
    tar_guidance_scale: float = 6.0,
    n_min: int = 0,
    n_max: int = 20,
    max_sequence_length: int = 512,
    scale_mode: str = "slider",
    normalize_v_dir: bool = False,
    v_dir_target_norm: float = 1.0,
    log_vectors: bool = False,
    log_output_dir: Optional[str] = None,
):
    """
    FlowSlider for Z-Image with 3-prompt directional decomposition.

    Uses 6-way CFG with list-based processing (Z-Image specific):
    - [src_uncond, src_cond, tar_pos_uncond, tar_pos_cond, tar_neg_uncond, tar_neg_cond]

    Args:
        pipe: ZImagePipeline
        scheduler: Scheduler (typically FlowMatchEulerDiscreteScheduler)
        x_src: Source image latent (B, C, H, W)
        src_prompt: Source prompt describing the original image
        tar_prompt: Positive target prompt (e.g., "a severely decayed building")
        tar_prompt_neg: Negative target prompt (e.g., "a new building")
        negative_prompt: Negative prompt for CFG (usually empty)
        strength: Edit intensity strength (0.0 = tar_neg direction, 1.0 = tar_pos direction)
        T_steps: Total number of timesteps (default: 28 for Z-Image)
        n_avg: Number of velocity field averaging iterations
        src_guidance_scale: Guidance strength for source prompt (default: 2.0 for Z-Image)
        tar_guidance_scale: Guidance strength for target prompts (default: 6.0 for Z-Image)
        n_min: Number of final steps using regular sampling
        n_max: Maximum number of steps to apply flow editing (default: 20 for Z-Image)
        max_sequence_length: Maximum prompt token length (default: 512)
        scale_mode: Scaling method - "slider" (default), "interp", "step", "cfg", or "direct"
            - "slider": Scale the direction vector V_delta_s = V_fid + strength * V_steer
            - "direct": Scale the full velocity difference V_delta_s = strength * (V_pos - V_src) without decomposition
        normalize_v_dir: If True, normalize V_steer to v_dir_target_norm before scaling
        v_dir_target_norm: Target L2 norm for V_steer normalization
        log_vectors: If True, record vector statistics
        log_output_dir: Output directory for vector logs

    Returns:
        Edited latent tensor
    """
    from FlowEdit_utils import calculate_shift

    # Validate log_vectors arguments
    if log_vectors and log_output_dir is None:
        raise ValueError("log_output_dir must be specified when log_vectors=True")

    # Initialize logging variables
    stats_list = [] if log_vectors else None
    prev_V_steer = None
    prev_zt_edit = None

    device = x_src.device

    # ============================================
    # Timestep preparation (Z-Image specific)
    # ============================================
    image_seq_len = (x_src.shape[2] // 2) * (x_src.shape[3] // 2)

    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        sigmas=None,
        mu=mu,
    )

    # ============================================
    # Encode prompts (3 prompts with CFG)
    # Z-Image returns List[Tensor] format
    # ============================================

    # Source prompt
    src_prompt_embeds, src_negative_prompt_embeds = pipe.encode_prompt(
        prompt=src_prompt,
        device=device,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        max_sequence_length=max_sequence_length,
    )

    # Target positive prompt
    tar_pos_prompt_embeds, tar_pos_negative_prompt_embeds = pipe.encode_prompt(
        prompt=tar_prompt,
        device=device,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        max_sequence_length=max_sequence_length,
    )

    # Target negative prompt
    tar_neg_prompt_embeds, tar_neg_negative_prompt_embeds = pipe.encode_prompt(
        prompt=tar_prompt_neg,
        device=device,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        max_sequence_length=max_sequence_length,
    )

    # Extract embeddings from list format
    src_neg_emb = src_negative_prompt_embeds[0] if isinstance(src_negative_prompt_embeds, list) else src_negative_prompt_embeds
    src_pos_emb = src_prompt_embeds[0] if isinstance(src_prompt_embeds, list) else src_prompt_embeds
    tar_pos_neg_emb = tar_pos_negative_prompt_embeds[0] if isinstance(tar_pos_negative_prompt_embeds, list) else tar_pos_negative_prompt_embeds
    tar_pos_pos_emb = tar_pos_prompt_embeds[0] if isinstance(tar_pos_prompt_embeds, list) else tar_pos_prompt_embeds
    tar_neg_neg_emb = tar_neg_negative_prompt_embeds[0] if isinstance(tar_neg_negative_prompt_embeds, list) else tar_neg_negative_prompt_embeds
    tar_neg_pos_emb = tar_neg_prompt_embeds[0] if isinstance(tar_neg_prompt_embeds, list) else tar_neg_prompt_embeds

    # 6-way prompt embeddings list:
    # [src_uncond, src_cond, tar_pos_uncond, tar_pos_cond, tar_neg_uncond, tar_neg_cond]
    prompt_embeds_list = [
        src_neg_emb,      # src_uncond
        src_pos_emb,      # src_cond
        tar_pos_neg_emb,  # tar_pos_uncond
        tar_pos_pos_emb,  # tar_pos_cond
        tar_neg_neg_emb,  # tar_neg_uncond
        tar_neg_pos_emb,  # tar_neg_cond
    ]

    # Initialize ODE: zt_edit = x_src
    zt_edit = x_src.clone()

    # ============================================
    # Main editing loop
    # ============================================
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc=f"ZImage-FlowSlider (strength={strength:.2f})"):

        if T_steps - i > n_max:
            continue

        # Get timestep values from scheduler sigmas
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if scheduler.step_index + 1 < len(scheduler.sigmas):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = torch.zeros_like(t_i)

        if T_steps - i > n_min:
            # Flow-based editing phase

            V_delta_s_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                # Forward noise
                fwd_noise = torch.randn_like(x_src).to(device)

                # Source trajectory
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise

                # Target trajectory (with offset preservation)
                zt_tar = zt_edit + zt_src - x_src

                # ============================================
                # 6-way CFG with list-based processing (Z-Image specific)
                # ============================================
                # Prepare latents list: [src_uncond, src_cond, tar_pos_uncond, tar_pos_cond, tar_neg_uncond, tar_neg_cond]
                # Z-Image expects List[(C, 1, H, W)] format
                transformer_dtype = pipe.transformer.dtype
                latents_list = [
                    zt_src.squeeze(0).unsqueeze(1).to(transformer_dtype),  # src_uncond
                    zt_src.squeeze(0).unsqueeze(1).to(transformer_dtype),  # src_cond
                    zt_tar.squeeze(0).unsqueeze(1).to(transformer_dtype),  # tar_pos_uncond
                    zt_tar.squeeze(0).unsqueeze(1).to(transformer_dtype),  # tar_pos_cond
                    zt_tar.squeeze(0).unsqueeze(1).to(transformer_dtype),  # tar_neg_uncond
                    zt_tar.squeeze(0).unsqueeze(1).to(transformer_dtype),  # tar_neg_cond
                ]

                # Z-Image timestep format: (1000 - t) / 1000
                timestep_zimage = (1000 - t) / 1000
                timestep_batch = timestep_zimage.expand(len(latents_list))

                # Single transformer call with list input
                with torch.no_grad():
                    noise_pred_list = pipe.transformer(
                        latents_list,
                        timestep_batch,
                        prompt_embeds_list,
                        return_dict=False,
                    )[0]

                # Apply sign inversion and squeeze frame dimension (Z-Image specific)
                noise_pred_list = [-pred.squeeze(1) for pred in noise_pred_list]

                # Split into 6 predictions
                (
                    src_noise_uncond, src_noise_cond,
                    tar_pos_noise_uncond, tar_pos_noise_cond,
                    tar_neg_noise_uncond, tar_neg_noise_cond,
                ) = noise_pred_list

                # Apply CFG to get 3 velocities
                Vt_src = src_noise_uncond + src_guidance_scale * (src_noise_cond - src_noise_uncond)
                Vt_pos = tar_pos_noise_uncond + tar_guidance_scale * (tar_pos_noise_cond - tar_pos_noise_uncond)
                Vt_neg = tar_neg_noise_uncond + tar_guidance_scale * (tar_neg_noise_cond - tar_neg_noise_uncond)

                # ============================================
                # Directional decomposition (same as FLUX/SD3)
                # ============================================
                V_steer = Vt_pos - Vt_neg
                V_fid = Vt_neg - Vt_src

                # V_delta_s computation depends on scale_mode
                if scale_mode == "slider":
                    if normalize_v_dir:
                        v_dir_norm = V_steer.norm(dim=-1, keepdim=True).mean()
                        V_steer_scaled = V_steer * (v_dir_target_norm / (v_dir_norm + 1e-8))
                    else:
                        V_steer_scaled = V_steer
                    V_delta_s = V_fid + strength * V_steer_scaled
                elif scale_mode == "direct":
                    # Direct mode: strength the full velocity difference without decomposition
                    V_delta_full = Vt_pos - Vt_src
                    V_delta_s = strength * V_delta_full
                elif scale_mode == "interp":
                    V_delta_full = Vt_pos - Vt_src
                    V_final = Vt_src + strength * V_delta_full
                    V_delta_s = V_final
                elif scale_mode == "step":
                    V_delta_s = V_fid + V_steer
                elif scale_mode == "cfg":
                    V_delta_s = V_fid + V_steer
                else:
                    raise ValueError(f"Unknown scale_mode: {scale_mode}")

                # Add batch dimension back for accumulation
                V_delta_s_avg += (1 / n_avg) * V_delta_s.unsqueeze(0)

            # ============================================
            # Vector Logging (if enabled)
            # ============================================
            if log_vectors:
                step_stats = compute_vector_stats(
                    V_fid=V_fid.unsqueeze(0),
                    V_steer=V_steer.unsqueeze(0),
                    V_delta_s=V_delta_s_avg,
                    zt_edit=zt_edit,
                    prev_V_steer=prev_V_steer,
                    prev_zt_edit=prev_zt_edit,
                )
                step_stats["timestep"] = t_i.item() if hasattr(t_i, 'item') else float(t_i)
                if normalize_v_dir and scale_mode == "slider":
                    step_stats["normalize_v_dir"] = True
                    step_stats["v_dir_target_norm"] = v_dir_target_norm
                    step_stats["v_dir_original_norm"] = V_steer.norm(dim=-1).mean().item()
                stats_list.append(step_stats)
                prev_V_steer = V_steer.unsqueeze(0).clone()
                prev_zt_edit = zt_edit.clone()

            # Propagate ODE
            zt_edit = zt_edit.to(torch.float32)
            if scale_mode == "step":
                zt_edit = zt_edit + strength * (t_im1 - t_i) * V_delta_s_avg
            else:
                zt_edit = zt_edit + (t_im1 - t_i) * V_delta_s_avg
            zt_edit = zt_edit.to(V_delta_s_avg.dtype)

        else:  # Regular sampling for last n_min steps
            if i == T_steps - n_min:
                # Initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            # For final steps, use interpolated target embedding
            interp_emb = (1 - strength) * tar_neg_pos_emb + strength * tar_pos_pos_emb

            # 2-way CFG for interpolated target
            transformer_dtype = pipe.transformer.dtype
            latents_list = [
                xt_tar.squeeze(0).unsqueeze(1).to(transformer_dtype),  # uncond
                xt_tar.squeeze(0).unsqueeze(1).to(transformer_dtype),  # cond
            ]
            prompt_embeds_2way = [tar_pos_neg_emb, interp_emb]

            timestep_zimage = (1000 - t) / 1000
            timestep_batch = timestep_zimage.expand(2)

            with torch.no_grad():
                noise_pred_list = pipe.transformer(
                    latents_list,
                    timestep_batch,
                    prompt_embeds_2way,
                    return_dict=False,
                )[0]

            # Apply sign inversion and squeeze
            noise_pred_list = [-pred.squeeze(1) for pred in noise_pred_list]
            interp_uncond, interp_cond = noise_pred_list

            Vt_tar = interp_uncond + tar_guidance_scale * (interp_cond - interp_uncond)

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar.unsqueeze(0)
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    out = zt_edit if n_min == 0 else xt_tar

    # ============================================
    # Save and visualize vector statistics
    # ============================================
    if log_vectors and stats_list:
        stats_path = save_vector_stats(stats_list, log_output_dir, strength)
        plot_vector_stats(stats_path, log_output_dir)
        print(f"Vector statistics saved to {log_output_dir}")

    return out
