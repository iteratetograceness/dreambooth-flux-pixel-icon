from dataclasses import dataclass
from pathlib import Path
import itertools
from modal import App, Image, gpu, Secret, Volume, enter, method

FOLDER_1 = "lr_1e-06_steps_4000_rank_8"
FOLDER_2 = "lr_0.0002_steps_4000_rank_16"
FOLDER_3 = "lr_0.0002_steps_4200_rank_16"

DATASET_1 = "graceyun/pixel-pngs-dreambooth" # 31 images
DATASET_2 = "graceyun/dreambooth-pixels" # 42 images

app = App(name="dreambooth-flux")

GIT_SHA = "7fb481f840b5d73982cafd1affe89f21a5c0b20b"
volume = Volume.from_name(
    "dreambooth-flux", create_if_missing=True
)
MODEL_DIR = "/dreambooth-flux"
VOLUME_CONFIG = { MODEL_DIR: volume }

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "datasets",
        "bitsandbytes",
        "hf_transfer",
        "wandb",
        "torch"
    )
    .apt_install("git")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
        "cd /root/examples/dreambooth && pip install -r requirements_flux.txt"
    )
)

@dataclass
class TrainConfig():
    instance_name: str = "PXCON"
    # class_name: str = "a simple 16-bit pixel art icon on a white background"
    model_name: str = "black-forest-labs/FLUX.1-dev"
    dataset_name: str = DATASET_2
    resolution: int = 512
    train_batch_size: int = 42
    gradient_accumulation_steps: int = 1
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    checkpointing_steps: int = 1000
    validation_prompt: str = "a PXCON, a 16-bit pixel art icon of a brown puppy, on a white background"
    
@dataclass
class SweepConfig():
    learning_rates = [2e-4] # 1e-6, 8e-5, 2e-4
    train_steps = [4200] # [1000, 1500, 3000, 4000]
    ranks = [16] # [4, 8, 16]

def generate_sweep_configs(sweep_config: SweepConfig):
    param_combinations = itertools.product(
        sweep_config.learning_rates,
        sweep_config.train_steps,
        sweep_config.ranks,
    )

    return [
        {
            "learning_rate": lr,
            "max_train_steps": steps,
            "rank": rank,
            "output_dir": f"{MODEL_DIR}/lr_{lr}_steps_{steps}_rank_{rank}",
        }
        for lr, steps, rank in param_combinations
    ]

huggingface_secret = Secret.from_name(
    "huggingface-secret"
)
wandb_secret = Secret.from_name(
    "wandb_api_token"
)
image = image.env(
    { 
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    }
)

@app.function(
    image=image,
    gpu=gpu.H100(),
    volumes=VOLUME_CONFIG,
    timeout=24 * 60 * 60,
    secrets=[huggingface_secret, wandb_secret],
    memory=102400,
)
def train(config):
    import subprocess
    from accelerate.utils import write_basic_config

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="bf16")

    # the model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate
    def _exec_subprocess(cmd: list[str]):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    print("Launching Dreambooth training script")
    train_config = TrainConfig()
    output_dir = config['output_dir']
    learning_rate = config['learning_rate']
    max_train_steps = config['max_train_steps']
    rank = config['rank']
    
    instance_prompt = f"a {train_config.instance_name}, a 16-bit pixel art icon in a minimalist style, on a white background"

    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "examples/dreambooth/train_dreambooth_lora_flux.py",
            "--mixed_precision=bf16",
            f"--pretrained_model_name_or_path={train_config.model_name}",
            f"--dataset_name={train_config.dataset_name}",
            f"--output_dir={output_dir}",
            f"--instance_prompt={instance_prompt}",
            f"--resolution={train_config.resolution}",
            f"--train_batch_size={train_config.train_batch_size}",
            f"--gradient_accumulation_steps={train_config.gradient_accumulation_steps}",
            f"--learning_rate={learning_rate}",
            f"--lr_scheduler={train_config.lr_scheduler}",
            f"--lr_warmup_steps={train_config.lr_warmup_steps}",
            f"--max_train_steps={max_train_steps}",
            f"--checkpointing_steps={train_config.checkpointing_steps}",
            f"--rank={rank}",
            "--caption_column=text",
            # f"--validation_prompt={train_config.validation_prompt}",
            "--report_to=wandb",
            "--push_to_hub",
            "--gradient_checkpointing",
            "--use_8bit_adam",
            "--center_crop"
        ]
    )
    volume.commit()

@app.local_entrypoint()
def run():    
    print("🎨 Training model")
    sweep_config = SweepConfig()
    configs = generate_sweep_configs(sweep_config)
    
    for config in configs:
        train.remote(config)

    print("🎨 Training finished")
    
    # example_prompts = [
    #     "a PXCON, a 16-bit pixel art icon of an hourglass",
    #     "a PXCON, a 16-bit pixel art icon of a macintosh computer",
    #     "a PXCON, a 16-bit pixel art icon of a cute bunny"
    # ]
    
    # model = Model()
    # config = InferenceConfig()
    
    # for prompt in example_prompts:
    #     model.inference.remote(prompt, config)
    #     print(f"Saved generated image for prompt: {prompt}")
        