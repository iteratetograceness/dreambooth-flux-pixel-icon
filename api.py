import base64
from dataclasses import dataclass
from fastapi import FastAPI, Response, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from modal import App, Image, Volume, Secret, gpu, enter, method, asgi_app, parameter
from pydantic import BaseModel
import io

# MODEL VOLUME CONFIG
volume = Volume.from_name(
    "dreambooth-flux"
)
MODEL_DIR = "/dreambooth-flux"
VOLUME_CONFIG = { MODEL_DIR: volume }
PIXEL_COLOR_DIR = "lr_0.0002_steps_4000_rank_16"

# STYLE CONFIG
STYLE_TO_DIR = {
    "color": PIXEL_COLOR_DIR,
}
DEFAULT_STYLE = "color"

# BASE CUDA IMAGE FOR GPU SUPPORT
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
base_cuda_image = Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

# WEB APP IMAGE
web_app_image = Image.debian_slim().pip_install(
    "fastapi[standard]",
    "pydantic",
)

# MODEL IMAGE
diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"
model_image = (
    base_cuda_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark>=0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]>=0.26.2",
        "accelerate==0.33.0",
        "safetensors>=0.4.4",
        "sentencepiece>=0.2.0",
        "torch==2.5.0",
        "peft",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_CACHE": "/cache",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    })
)

# flux_image = (
#     cuda_dev_image.apt_install(
#         "git",
#         "libglib2.0-0",
#         "libsm6",
#         "libxrender1",
#         "libxext6",
#         "ffmpeg",
#         "libgl1",
#     )
#     .pip_install(
#         "boto3",
#         "fastapi[standard]",
#         "invisible_watermark==0.2.0",
#         "transformers==4.44.0",
#         "huggingface_hub[hf_transfer]==0.26.2",
#         "accelerate==0.33.0",
#         "safetensors==0.4.4",
#         "sentencepiece==0.2.0",
#         "torch==2.5.0",
#         "peft",
#         f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
#         "numpy<2",
#     )
#     .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
# )

# flux_image = flux_image.env(
#     {
#         "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
#         "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
#     }
# )

web_app = FastAPI(title="Pixel API", description="Generate pixel icons")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://dotelier.studio",
        "https://www.dotelier.studio",
    ],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Proxy-Authorization"],
)

app = App(name="pixel-api")

huggingface_secret = Secret.from_name(
    "huggingface-secret"
)

# MODEL INPUT CONFIG
class InputModel(BaseModel):
    prompt: str
    num_outputs: int = 1
    seed: int | None = None
class InferenceConfig(BaseModel):
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_outputs: int = 1
    seed: int | None = None

# Hot-swappable LoRA weights via a different method (e.g. load_weights)?For later: when b&w LoRA is ready.

# MODEL CLASS
@app.cls(
    gpu=gpu.H100(),
    image=model_image,
    volumes={
        **VOLUME_CONFIG,
         "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": Volume.from_name("inductor-cache", create_if_missing=True),
    },
    secrets=[huggingface_secret],
    container_idle_timeout=60 * 60,
    timeout=60 * 60,
    allow_concurrent_inputs=10,
    concurrency_limit=5,
    memory=32768,
)
class PixelColorModel:
    # Add a parameter to determine which weights to load
    style: str = parameter(default=DEFAULT_STYLE)
    compile: int = parameter(default=0)
    
    @enter()
    def load_model(self):
        try:
            print(f"🔥 Starting model load with style: {self.style}")
            
            import torch
            from diffusers import AutoPipelineForText2Image
            
            volume.reload()
                        
            print("🤗 Loading pipeline...")
            pipeline = AutoPipelineForText2Image.from_pretrained(
                "black-forest-labs/FLUX.1-dev", 
                torch_dtype=torch.bfloat16,
            ).to('cuda')
            
            print("⚡ Optimizing pipeline...")
            self.pipeline = optimize(pipeline, compile=bool(self.compile))
               
            dir = f"{MODEL_DIR}/{STYLE_TO_DIR[self.style]}"
            print(f"🎯 Loading LoRA weights from {dir}...")
            self.pipeline.load_lora_weights(dir, weight_name='pytorch_lora_weights.safetensors')
            
            print("✨ Model load complete!")
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

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

# WEB APP IMPLEMENTATION
def make_error_readable(error):
    return f"Field '{' -> '.join(error['loc'])}': {error['msg']}"

@web_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    readable_errors = [make_error_readable(error) for error in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({ "detail": readable_errors }),
    )

@web_app.post("/v1/pixel-color")
async def generate_pixel_color(input: InputModel):
    try:
        prompt = f"a PXCON, a 16-bit pixel art icon of {input.prompt}"
        print(f"🔦 Starting Pixel Colorinference with prompt: {prompt}")
        
        images = pixel_model.inference.remote(
            text=prompt,
            config=InferenceConfig(
                num_outputs=input.num_outputs,
                seed=input.seed
            ),
        )
        
        base64_images = [base64.b64encode(img).decode('utf-8') for img in images]
        print(f"🎉 Pixel Color inference completed, got {len(images)} images!")
        
        return JSONResponse(content={"images": base64_images})
    except Exception as e:
        print(f"Error in generate_pixel_color: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

@app.function(
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=1800,
    memory=8192
)
@asgi_app(label="pixel-api")
def fastapi_app():
    try:
        print("🔄 Starting FastAPI app initialization...")
        
        def init_models():
            global pixel_model
            pixel_model = PixelColorModel(compile=1)
        
        init_models()
        print("🎉 FastAPI app initialization complete!")
        return web_app
    except Exception as e:
        print(f"❌ Failed to initialize FastAPI app: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

# Source: https://modal.com/docs/examples/flux#run-flux-fast-on-h100s-with-torchcompile
# Specifics w/ LoRA: https://github.com/huggingface/diffusers/issues/9279
def optimize(pipe, compile=True):
    import torch
    
    print("🔄 Starting optimization process...")
    
    # Skip QKV fusion since it breaks LoRA   
    # fuse QKV projections in Transformer and VAE
    # pipe.transformer.fuse_qkv_projections()
    # pipe.vae.fuse_qkv_projections()

    # Switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # Set torch compile flags
    config = torch._inductor.config
    config.disable_progress = False
    config.conv_1x1_as_mm = True  # Treat 1x1 convolutions as matrix muls
    
    # Adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # Do not fuse pointwise ops into matmuls

    # Tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    print("🔦 Running torch compilation (may take up to 20 minutes)...")
    pipe(
        "test compilation with LoRA weights",
        output_type="pil",
        num_inference_steps=1,
    ).images[0]
    print("🔦 Finished torch compilation")
    
    print("✅ Model optimization complete with LoRA weights verified")

    return pipe
