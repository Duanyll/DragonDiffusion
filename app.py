from src.demo.download import download_all
# download_all()

from src.demo.demo import create_demo_move, create_demo_appearance, create_demo_drag, create_demo_face_drag, create_demo_paste
from src.demo.model import DragonModels

import cv2
import gradio as gr

import time
from torch.profiler import profile, record_function, ProfilerActivity

profiler_warmup = False
def wrap_profiler(func):
    def wrapper(*args, **kwargs):
        global profiler_warmup
        if profiler_warmup:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function(func.__name__):
                    result = func(*args, **kwargs)
            outfile = f"profile-{func.__name__}-{time.time()}.json"
            print(f"Saving profile to {outfile}")
            prof.export_chrome_trace(outfile)
            return result
        else:
            result = func(*args, **kwargs)
            profiler_warmup = True
            return result
    return wrapper

# main demo
pretrained_model_path = "pt-sk/stable-diffusion-1.5"
model = DragonModels(pretrained_model_path=pretrained_model_path)

DESCRIPTION = '# 🐉🐉[DragonDiffusion V1.0](https://github.com/MC-E/DragonDiffusion)🐉🐉'

DESCRIPTION += f'<p>Gradio demo for [DragonDiffusion](https://arxiv.org/abs/2307.02421) and [DiffEditor](https://arxiv.org/abs/2307.02421). If it is helpful, please help to recommend [[GitHub Repo]](https://github.com/MC-E/DragonDiffusion) to your friends 😊 </p>'

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Appearance Modulation'):
            create_demo_appearance(wrap_profiler(model.run_appearance))
        with gr.TabItem('Object Moving & Resizing'):
            create_demo_move(wrap_profiler(model.run_move))
        with gr.TabItem('Face Modulation'):
            create_demo_face_drag(wrap_profiler(model.run_drag_face))
        with gr.TabItem('Content Dragging'):
            create_demo_drag(wrap_profiler(model.run_drag))
        with gr.TabItem('Object Pasting'):
            create_demo_paste(wrap_profiler(model.run_paste))

demo.queue(concurrency_count=3, max_size=20)
demo.launch(server_name="0.0.0.0")
