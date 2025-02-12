from dataclasses import dataclass
from fastapi import FastAPI, Response, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from modal import App, Image, Volume, Secret, gpu, enter, method, asgi_app, parameter
from pydantic import BaseModel
import io

volume = Volume.from_name(
    "dreambooth-flux"
)
MODEL_DIR = "/dreambooth-flux"
VOLUME_CONFIG = { MODEL_DIR: volume }
PIXEL_COLOR_DIR = "lr_0.0002_steps_4000_rank_16"

cuda_version = "12.4.0"
flavor = "devel"  # Includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
cuda_dev_image = Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "boto3",
        "fastapi[standard]",
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.0",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)

flux_image = flux_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
)

web_app = FastAPI(title="Pixel API", description="Generate pixel icons")
app = App(name="pixel-api", image=flux_image)

huggingface_secret = Secret.from_name(
    "huggingface-secret"
)

class InputModel(BaseModel):
    prompt: str
    num_outputs: int = 1
    seed: int | None = None

@dataclass
class InferenceConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_outputs: int = 1
    seed: int | None = None

@app.cls(
    # image=image,
    gpu=gpu.H100(),
    volumes={
        **VOLUME_CONFIG,
         "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": Volume.from_name("inductor-cache", create_if_missing=True),
    },
    secrets=[huggingface_secret],
    container_idle_timeout=20 * 60,
    timeout=60 * 60,
)
class PixelColorModel:
    compile: int = parameter(default=0)
    
    @enter()
    def load_model(self):
        import torch
        from diffusers import FluxPipeline
        
        # Reload the modal.Volume to ensure the latest state is accessible:
        volume.reload()
        
        pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
        
        lora_path = f"{MODEL_DIR}/{PIXEL_COLOR_DIR}"
        pipeline.load_lora_weights(lora_path, weight_name='pytorch_lora_weights.safetensors')
        
        self.pipeline = optimize(pipeline, compile=bool(self.compile))

    @method()
    def inference(self, text: str, config: InferenceConfig):
        images = self.pipeline(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            num_images_per_prompt=config.num_outputs
        ).images
        
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        
        return image_output

## START WEB APP IMPLEMENTATION ##
def make_readable(error):
    return f"Field '{' -> '.join(error['loc'])}': {error['msg']}"

@web_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    readable_errors = [make_readable(error) for error in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({ "detail": readable_errors }),
    )

@web_app.post("/v1/pixel-color")
async def generate_pixel_color(input: InputModel):
    model = PixelColorModel(compile=1)
    prompt = f"a PXCON, a 16-bit pixel art icon of {input.prompt}"
    images = model.inference.remote(
        prompt=prompt,
        num_outputs=input.num_outputs,
        seed=input.seed
    )
    return Response(content=images[0], media_type="image/png")

@app.function(
    # image=image,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret]
)
@asgi_app(label="pixel-api")
def fastapi_app():
    return web_app
## END WEB APP IMPLEMENTATION ##

# Source: https://modal.com/docs/examples/flux#run-flux-fast-on-h100s-with-torchcompile
def optimize(pipe, compile=True):
    import torch
    
    # fuse QKV projections in Transformer and VAE
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # set torch compile flags
    config = torch._inductor.config
    config.disable_progress = False  # show progress bar
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    # tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    # trigger torch compilation
    print("🔦 Running torch compilation (may take up to 20 minutes)...")

    pipe(
        "a dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=50,
    ).images[0]

    print("🔦 Finished torch compilation")

    return pipe