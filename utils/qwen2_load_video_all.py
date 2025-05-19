import time
import torch
import numpy as np
from decord import VideoReader, cpu
from qwen_vl_utils.vision_process import (
    IMAGE_FACTOR,
    VIDEO_MIN_PIXELS,
    VIDEO_MAX_PIXELS,
    VIDEO_TOTAL_PIXELS,
    FRAME_FACTOR,
    smart_resize,
    extract_vision_info,
)
from torchvision.transforms.functional import resize, InterpolationMode
import logging

logger = logging.getLogger(__name__)

def _read_video_batch(ele: dict, frame_idx: list[int]) -> torch.Tensor:
    """
    Decode and stack frames at the given indices.
    """
    vr = VideoReader(ele["video"], ctx=cpu(0))
    # sort & truncate to actual number of indices
    idx = sorted(frame_idx)
    arr = vr.get_batch(idx).asnumpy()                             
    # to T,C,H,W
    return torch.tensor(arr).permute(0, 3, 1, 2)

def fetch_video_dynamic(ele: dict, max_tokens: int = 32768) -> torch.Tensor:
    """
    1. Compute per-frame token cost via pixel→token rule (28×28 px = 1 token). :contentReference[oaicite:0]{index=0}
    2. Determine how many frames we can afford: max_frames = max_tokens // tokens_per_frame.
    3. Uniformly sample that many frames via np.linspace.
    4. Decode & resize them.
    """
    # 1. Decode metadata
    vr = VideoReader(ele["video"], ctx=cpu(0))
    total_frames = len(vr)                                         
    # average frame resolution
    # we assume square-ish after resize, or approximate via VIDEO_TOTAL_PIXELS 
    avg_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS) / total_frames  
    # 2. Visual token cost per frame
    # Every 28×28 = 784 pixels → 1 token; min 4 tokens/frame :contentReference[oaicite:1]{index=1}
    tokens_per_frame = max(int(avg_pixels / (28 * 28)), 4)         
    max_frames = max_tokens // tokens_per_frame                     
    max_frames = min(max_frames, total_frames)                     
    logger.info(f"[dynamic] total={total_frames}, tokens/fr={tokens_per_frame}, max_frames={max_frames}")

    # 3. Uniform sampling
    idx = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

    # 4. Decode & preprocess
    video = _read_video_batch(ele, idx)
    T, C, H, W = video.shape

    # Compute resize bounds
    minp = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    totp = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    # derive a cap per-frame pixel budget
    cap = max(min(VIDEO_MAX_PIXELS, totp / T * FRAME_FACTOR),
              int(minp * 1.05))
    cap = ele.get("max_pixels", cap)

    # final resize dims
    if "resized_height" in ele and "resized_width" in ele:
        h, w = smart_resize(ele["resized_height"], ele["resized_width"], factor=IMAGE_FACTOR)
    else:
        h, w = smart_resize(H, W, factor=IMAGE_FACTOR, min_pixels=minp, max_pixels=cap)

    video = resize(video, [h, w], InterpolationMode.BICUBIC, antialias=True).float()
    return video

def process_vision_info_dynamic(
    convs: list[dict] | list[list[dict]],
    max_tokens: int = 32768
) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
    """
    Similar to process_vision_info_frame_idx, but automatically samples uniformly
    up to the dynamic max_frames budget.
    """
    infos = extract_vision_info(convs)                           
    imgs, vids = [], []
    for ele in infos:
        if "video" in ele:
            vids.append(fetch_video_dynamic(ele, max_tokens=max_tokens))
        else:
            raise ValueError("Only video elements supported")
    return (None if not imgs else imgs,
            None if not vids else vids)
