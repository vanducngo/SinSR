#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Yufei Wang

import argparse
import gradio as gr
from pathlib import Path

from omegaconf import OmegaConf
from sampler import Sampler

from utils import util_image
from basicsr.utils.download_util import load_file_from_url

def get_configs(model, colab):
    configs = None
    if model == 'SinSR':
        if colab:
            configs = OmegaConf.load('/content/SinSR/configs/SinSR.yaml')
        else:
            configs = OmegaConf.load('./configs/SinSR.yaml')
    elif model == 'ResShift':
        if colab:
            configs = OmegaConf.load('/content/SinSR/configs/realsr_swinunet_realesrgan256.yaml')
        else:
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
        task = "realsrx4"

    # prepare the checkpoint
    ckpt_dir = Path('./weights')
    ckpt_path = ''
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    if model == 'SinSR':
        ckpt_path = ckpt_dir / f'SinSR_v1.pth'
        if not ckpt_path.exists():
            print("Chưa có checkpoint trong SinSR => Download về")
            load_file_from_url(
                url=f"https://github.com/wyf0912/SinSR/releases/download/v1.0/{ckpt_path.name}",
                model_dir=ckpt_dir,
                progress=True,
                file_name=ckpt_path.name,
                )
    elif model == 'ResShift':
        ckpt_path = ckpt_dir / f'resshift_{task}_s15_v1.pth'
        if not ckpt_path.exists():
            print("Chưa có checkpoint trong ResShift => Download về")
            load_file_from_url(
                url=f"https://github.com/zsyOAOA/ResShift/releases/download/v2.0/{ckpt_path.name}",
                model_dir=ckpt_dir,
                progress=True,
                file_name=ckpt_path.name,
                )
    
    vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
         load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = 15
    configs.diffusion.params.sf = 4
    configs.autoencoder.ckpt_path = str(vqgan_path)

    return configs

def predict(in_path, colab = True, model='SinSR', seed=12345):
    configs = get_configs(model, colab)
    if sampler_dict[model] is None:
        sampler_dict[model] = Sampler(
            configs,
            chop_size=256,
            chop_stride=224,
            chop_bs=1,
            use_fp16=True,
            seed=seed,
        )
    
    sampler = sampler_dict[model]
    
    out_dir = Path('restored_output')
    if not out_dir.exists():
        out_dir.mkdir()
    
    single_step = True

    # suy luận (inference) bằng cách sử dụng mô hình đã huấn luyện trong SinSR
    sampler.inference(in_path, out_dir, one_step=single_step)

    out_path = out_dir / f"{Path(in_path).stem}.png"
    assert out_path.exists(), 'Super-resolution failed!'
    im_sr = util_image.imread(out_path, chn="rgb", dtype="uint8")

    return im_sr, str(out_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='SinSR: Diffusion-Based Image Super-Resolution in a Single Step')
    parser.add_argument('--colab', action='store_true', help = "Change paths to match colab path locations")
    
    args = parser.parse_args()
    
    sampler_dict = {"SinSR": None, "ResShift": None} 

    title = "SinSR: Diffusion-Based Image Super-Resolution in a Single Step"
    description = ""
    article = ''
    

    if args.colab:
        examples=[
            ['/content/SinSR/testdata/RealSet65/dog2.png', True, "SinSR", 12345],
            ['/content/SinSR/testdata/RealSet65/bears.jpg', True, "SinSR", 12345],
            ['/content/SinSR/testdata/RealSet65/oldphoto6.png', True, "SinSR", 12345],
          ]
    else:
        examples=[
            ['./testdata/RealSet65/dog2.png', True, "SinSR", 12345],
            ['./testdata/RealSet65/bears.jpg', True, "SinSR", 12345],
            ['./testdata/RealSet65/oldphoto6.png', True, "SinSR", 12345],
          ]
        
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="filepath", label="Input: Low Quality Image"),
            gr.Checkbox(label="Using colab?", value = True),
            gr.Dropdown(
                choices=["SinSR", "ResShift"],
                value="SinSR",
                label="Model",
                ),
            gr.Number(value=12345, precision=0, label="Random seed")
        ],
        outputs=[
            gr.Image(type="numpy", label="Output: High Quality Image"),
            gr.outputs.File(label="Download the output")
        ],
        title=title,
        description=description,
        article=article,
        examples = examples,
        allow_flagging="never"
        )

    demo.queue(concurrency_count=4)
    demo.launch(share=True)

