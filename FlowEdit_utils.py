from typing import Optional, Tuple, Union
import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


def resize_image_for_flux(
    image: Image.Image,
    max_short_edge: int = 1024,
) -> Tuple[Image.Image, bool]:
    """
    Resize image if short edge exceeds max_short_edge.
    Maintains aspect ratio and ensures dimensions are divisible by 16.

    Args:
        image: PIL Image to resize
        max_short_edge: Maximum size for shorter edge (default: 1024)

    Returns:
        Tuple of (resized_image, was_resized)
    """
    w, h = image.size
    short_edge = min(w, h)

    if short_edge <= max_short_edge:
        # Only ensure divisible by 16
        new_w = (w // 16) * 16
        new_h = (h // 16) * 16
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)
            return image, True
        return image, False

    # Calculate new dimensions maintaining aspect ratio
    scale = max_short_edge / short_edge
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Ensure divisible by 16
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16

    image_resized = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"  Resized for FLUX: {w}x{h} -> {new_w}x{new_h}")

    return image_resized, True


def load_and_resize_image(
    image_path: str,
    max_short_edge: int = 1024,
) -> Image.Image:
    """
    Load image and resize if necessary.

    Args:
        image_path: Path to image file
        max_short_edge: Maximum size for shorter edge

    Returns:
        PIL Image (resized if needed)
    """
    image = Image.open(image_path).convert("RGB")
    image, _ = resize_image_for_flux(image, max_short_edge)
    return image



def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar



def calc_v_zimage(pipe, latents_list, prompt_embeds_list, src_guidance_scale, tar_guidance_scale, t):
    """
    ZImage用の速度場計算

    Args:
        pipe: ZImagePipeline
        latents_list: List[Tensor] - [src_uncond, src_cond, tar_uncond, tar_cond] の4要素
        prompt_embeds_list: List[Tensor] - 対応するprompt embeddings
        src_guidance_scale: float - ソースプロンプトのCFGスケール
        tar_guidance_scale: float - ターゲットプロンプトのCFGスケール
        t: Tensor - タイムステップ (0-1000)

    Returns:
        noise_pred_src, noise_pred_tar: CFG適用後の速度場
    """
    # timestepを正規化 (ZImageは (1000-t)/1000 形式)
    timestep = (1000 - t) / 1000
    timestep = timestep.expand(len(latents_list))

    # latentsをList[Tensor]形式に変換
    # 入力: (C, H, W) -> 出力: (C, 1, H, W) でF(フレーム)次元を追加
    # transformerのdtypeに合わせる
    transformer_dtype = pipe.transformer.dtype
    latent_model_input_list = [lat.unsqueeze(1).to(transformer_dtype) for lat in latents_list]

    with torch.no_grad():
        # transformer forward
        noise_pred_list = pipe.transformer(
            latent_model_input_list,
            timestep,
            prompt_embeds_list,
            return_dict=False,
        )[0]

        # squeeze(1)でF次元を戻し、符号反転（ZImageの仕様）
        # 出力: (C, 1, H, W) -> (C, H, W)
        noise_pred_list = [-pred.squeeze(1) for pred in noise_pred_list]

        # CFG適用: [src_uncond, src_cond, tar_uncond, tar_cond]
        src_noise_pred_uncond = noise_pred_list[0]
        src_noise_pred_cond = noise_pred_list[1]
        tar_noise_pred_uncond = noise_pred_list[2]
        tar_noise_pred_cond = noise_pred_list[3]

        noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_cond - src_noise_pred_uncond)
        noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_cond - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred



@torch.no_grad()
def FlowEditSD3(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,):
    
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    
    # src prompts
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

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
 
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar) 

                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling for last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(noise_pred_tar.dtype)

            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar



@torch.no_grad()
def FlowEditFLUX(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,):

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

    x_src, latent_src_image_ids = pipe.prepare_latents(batch_size= x_src.shape[0], num_channels_latents=num_channels_latents, height=orig_height, width=orig_width, dtype=x_src.dtype, device=x_src.device, generator=None,latents=x_src)
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    # 5. Prepare timesteps
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

    
    # src prompts
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,

    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src_packed.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):
                                    

                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                
                zt_src = (1-t_i)*x_src_packed + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                # Merge in the future to avoid double computation
                Vt_src = calc_v_flux(pipe,
                                                    latents=zt_src,
                                                    prompt_embeds=src_prompt_embeds, 
                                                    pooled_prompt_embeds=src_pooled_prompt_embeds, 
                                                    guidance=src_guidance,
                                                    text_ids=src_text_ids, 
                                                    latent_image_ids=latent_src_image_ids, 
                                                    t=t)
                
                Vt_tar = calc_v_flux(pipe,
                                                    latents=zt_tar,
                                                    prompt_embeds=tar_prompt_embeds, 
                                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                                    guidance=tar_guidance,
                                                    text_ids=tar_text_ids, 
                                                    latent_image_ids=latent_tar_image_ids, 
                                                    t=t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed
                
            Vt_tar = calc_v_flux(pipe,
                                    latents=xt_tar,
                                    prompt_embeds=tar_prompt_embeds, 
                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                    guidance=tar_guidance,
                                    text_ids=tar_text_ids, 
                                    latent_image_ids=latent_tar_image_ids, 
                                    t=t)


            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out


@torch.no_grad()
def FlowEditZImage(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
    max_sequence_length: int = 512,):
    """
    ZImage用のFlowEdit実装

    Args:
        pipe: ZImagePipeline
        scheduler: FlowMatchEulerDiscreteScheduler
        x_src: Tensor - ソース画像のlatent (B, C, H, W)
        src_prompt: str - ソースプロンプト
        tar_prompt: str - ターゲットプロンプト
        negative_prompt: str - ネガティブプロンプト
        T_steps: int - 総ステップ数
        n_avg: int - 速度場の平均化回数
        src_guidance_scale: float - ソースCFGスケール
        tar_guidance_scale: float - ターゲットCFGスケール
        n_min: int - 通常サンプリングに切り替える最終ステップ数
        n_max: int - Flow編集を適用する最大ステップ数
        max_sequence_length: int - プロンプトの最大シーケンス長

    Returns:
        Tensor - 編集後のlatent
    """
    device = x_src.device

    # timestep準備（ZImageはcalculate_shiftを使用）
    height = x_src.shape[2] * pipe.vae_scale_factor * 2
    width = x_src.shape[3] * pipe.vae_scale_factor * 2
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

    # プロンプトエンコード
    # ソースプロンプト
    src_prompt_embeds, src_negative_prompt_embeds = pipe.encode_prompt(
        prompt=src_prompt,
        device=device,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        max_sequence_length=max_sequence_length,
    )

    # ターゲットプロンプト
    tar_prompt_embeds, tar_negative_prompt_embeds = pipe.encode_prompt(
        prompt=tar_prompt,
        device=device,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        max_sequence_length=max_sequence_length,
    )

    # prompt_embeds_list: [src_uncond, src_cond, tar_uncond, tar_cond]
    # ZImageのencode_promptはList[Tensor]を返すので、要素を取り出す
    src_neg_emb = src_negative_prompt_embeds[0] if isinstance(src_negative_prompt_embeds, list) else src_negative_prompt_embeds
    src_pos_emb = src_prompt_embeds[0] if isinstance(src_prompt_embeds, list) else src_prompt_embeds
    tar_neg_emb = tar_negative_prompt_embeds[0] if isinstance(tar_negative_prompt_embeds, list) else tar_negative_prompt_embeds
    tar_pos_emb = tar_prompt_embeds[0] if isinstance(tar_prompt_embeds, list) else tar_prompt_embeds

    prompt_embeds_list = [src_neg_emb, src_pos_emb, tar_neg_emb, tar_pos_emb]

    # initialize ODE: zt_edit = x_src
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):

        if T_steps - i > n_max:
            continue

        # タイムステップの計算
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if scheduler.step_index + 1 < len(scheduler.sigmas):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = torch.zeros_like(t_i)

        if T_steps - i > n_min:
            # Flow-based editing phase

            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                # ランダムノイズ
                fwd_noise = torch.randn_like(x_src).to(device)

                # 順方向プロセス: ソース軌道
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise

                # ターゲット軌道（オフセット維持）
                zt_tar = zt_edit + zt_src - x_src

                # latents_list: [src_uncond, src_cond, tar_uncond, tar_cond]
                latents_list = [zt_src.squeeze(0), zt_src.squeeze(0), zt_tar.squeeze(0), zt_tar.squeeze(0)]

                # 速度場計算
                Vt_src, Vt_tar = calc_v_zimage(
                    pipe,
                    latents_list,
                    prompt_embeds_list,
                    src_guidance_scale,
                    tar_guidance_scale,
                    t
                )

                # 速度場の差分を蓄積
                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src).unsqueeze(0)

            # ODE更新
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:  # 通常サンプリング（最後のn_minステップ）

            if i == T_steps - n_min:
                # SDEDIT-style generation phaseの初期化
                fwd_noise = torch.randn_like(x_src).to(device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            # ターゲットのみで速度場計算
            latents_list = [xt_tar.squeeze(0), xt_tar.squeeze(0), xt_tar.squeeze(0), xt_tar.squeeze(0)]

            _, Vt_tar = calc_v_zimage(
                pipe,
                latents_list,
                prompt_embeds_list,
                src_guidance_scale,
                tar_guidance_scale,
                t
            )

            # ODE更新
            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar.unsqueeze(0)
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    return zt_edit if n_min == 0 else xt_tar
