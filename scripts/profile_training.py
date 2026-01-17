#!/usr/bin/env python3
"""
Profile the VLA training pipeline to identify bottlenecks.
Run with: pixi run python scripts/profile_training.py
"""

import os
import sys
import time
from pathlib import Path

# Add workspace root
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf


def profile_step(name, func, *args, **kwargs):
    """Profile a single function call."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed*1000:.1f}ms")
    return result, elapsed


def main():
    print("=" * 60)
    print("VLA Training Pipeline Profiler")
    print("=" * 60)
    
    # Load config
    cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    cfg.framework.qwenvl.base_vlm = "Qwen/Qwen3-VL-2B-Instruct"
    
    # ============================================
    # 1. Profile Model Loading
    # ============================================
    print("\n[1] Model Loading:")
    from starVLA.model.framework import build_framework
    
    start = time.perf_counter()
    model = build_framework(cfg)
    model = model.cuda()
    model.eval()
    torch.cuda.synchronize()
    print(f"  Model load + move to GPU: {(time.perf_counter()-start)*1000:.1f}ms")
    
    # ============================================
    # 2. Create Fake Batch (simulating dataloader)
    # ============================================
    print("\n[2] Creating fake batch:")
    batch_size = 4
    
    # Create fake images like the dataloader would
    fake_images = []
    for _ in range(batch_size):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        fake_images.append([img])  # List of images per sample
    
    fake_batch = []
    for i in range(batch_size):
        sample = {
            "image": fake_images[i],
            "lang": "Pick up the red block and place it on the table.",
            "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
        }
        fake_batch.append(sample)
    
    print(f"  Created {batch_size} samples")
    
    # ============================================
    # 3. Profile the Forward Pass Components
    # ============================================
    print("\n[3] Forward Pass Breakdown:")
    
    # Get the VLM interface from model
    qwen_vl = model.qwen_vl_interface
    
    # Profile build_qwenvl_inputs
    batch_images = [sample["image"] for sample in fake_batch]
    instructions = [sample["lang"] for sample in fake_batch]
    
    start = time.perf_counter()
    qwen_inputs = qwen_vl.build_qwenvl_inputs(images=batch_images, instructions=instructions)
    build_inputs_time = (time.perf_counter() - start) * 1000
    print(f"  build_qwenvl_inputs: {build_inputs_time:.1f}ms")
    
    # Move inputs to GPU and profile
    start = time.perf_counter()
    for k, v in qwen_inputs.items():
        if isinstance(v, torch.Tensor):
            qwen_inputs[k] = v.cuda()
    torch.cuda.synchronize()
    move_to_gpu_time = (time.perf_counter() - start) * 1000
    print(f"  Move inputs to GPU: {move_to_gpu_time:.1f}ms")
    
    # Profile VLM forward
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = qwen_vl(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        vlm_forward_time = (time.perf_counter() - start) * 1000
        print(f"  VLM forward pass: {vlm_forward_time:.1f}ms")
    
    # Profile action model if exists
    if hasattr(model, 'action_model'):
        last_hidden = outputs.hidden_states[-1]
        actions = torch.randn(batch_size, 8, 7, device='cuda', dtype=torch.float32)
        
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
            torch.cuda.synchronize()
            start = time.perf_counter()
            action_loss = model.action_model(last_hidden, actions, None)
            torch.cuda.synchronize()
            action_time = (time.perf_counter() - start) * 1000
            print(f"  Action model forward: {action_time:.1f}ms")
    
    # ============================================
    # 4. Profile Full Forward Pass
    # ============================================
    print("\n[4] Full model.forward() (with gradients):")
    model.train()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_dict = model.forward(fake_batch)
    torch.cuda.synchronize()
    full_forward_time = (time.perf_counter() - start) * 1000
    print(f"  Full forward: {full_forward_time:.1f}ms")
    
    # ============================================
    # 5. Summary
    # ============================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  build_qwenvl_inputs:  {build_inputs_time:.1f}ms")
    print(f"  Move to GPU:          {move_to_gpu_time:.1f}ms")
    print(f"  VLM forward:          {vlm_forward_time:.1f}ms")
    print(f"  Full forward (train): {full_forward_time:.1f}ms")
    
    total_expected = build_inputs_time + move_to_gpu_time + vlm_forward_time
    print(f"\n  Expected total: ~{total_expected:.1f}ms")
    print(f"  If training takes 17000ms, the extra time is in:")
    print(f"    - Backward pass")
    print(f"    - Optimizer step")
    print(f"    - DeepSpeed overhead")
    print(f"    - Or repeated forward passes (gradient accumulation)")


if __name__ == "__main__":
    main()
