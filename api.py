import base64
from concurrent.futures import ThreadPoolExecutor
from modal import App, Image, Volume, Secret, gpu, enter, method, asgi_app
from pydantic import BaseModel
import io

# VOLUMES + CONFIG
volume = Volume.from_name("dreambooth-flux")
MODEL_DIR = "/dreambooth-flux"
STYLE_CONFIGS = {
    # "color_v1": {
    #     "dir": "lr_0.0002_steps_4000_rank_16",
    #     "prompt_prefix": "a PXCON, a 16-bit pixel art icon of "
    # },
    "color_v2": {
        "dir": "lr_0.0002_steps_4200_rank_16",
        "prompt_prefix": "a PXCON, a 16-bit pixel art icon of "
    }
}
DEFAULT_STYLE = "color_v2"
huggingface_secret = Secret.from_name("huggingface-secret")
auth_token = Secret.from_name("pixel-auth-token")
class InputModel(BaseModel):
    prompt: str
    num_outputs: int = 1
    seed: int | None = None
    style: str = DEFAULT_STYLE
class BodyModel(BaseModel):
    input: InputModel
class InferenceConfig(BaseModel):
    style: str = DEFAULT_STYLE
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_outputs: int = 1
    seed: int | None = None

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "peft",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
        "pydantic>=2.0",
        "sentencepiece>=0.1.91,!=0.1.92",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_CACHE": "/cache",
    })
)
app = App(name="pixel-api")
with image.imports():
    import torch
    from diffusers import AutoPipelineForText2Image

@app.cls(
    gpu=gpu.H100(),
    image=image,
    volumes={
        MODEL_DIR: volume,
        "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
    secrets=[huggingface_secret, auth_token],
    container_idle_timeout=1200,
    timeout=60 * 60,
    allow_concurrent_inputs=1,
    concurrency_limit=5,
    memory=32768,
)
class PixelModel:     
    @enter()
    def load_model(self):
        try:
            print(f"🤗 Starting model load...\n")
            
            self.pipelines = {}

            def load_lora(style: str):
                try:
                    print(f"🎯 Loading LoRA weights for {style}...")
                    pipeline = AutoPipelineForText2Image.from_pretrained(
                            "black-forest-labs/FLUX.1-dev", 
                            torch_dtype=torch.bfloat16,
                            add_prefix_space=False
                    ).to('cuda')
                    dir = f"{MODEL_DIR}/{STYLE_CONFIGS[style]['dir']}"
                    pipeline.load_lora_weights(dir, weight_name='pytorch_lora_weights.safetensors')
                    
                    # print("⚡ Compiling transformer...")
                    # pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune")
                    # _ = pipeline(
                    #     "warmup",
                    #     num_inference_steps=1,
                    #     num_images_per_prompt=1
                    # )
                    
                    self.pipelines[style] = pipeline
                    print(f"✨ LoRA weights loaded + compiled for {style}!")
                except Exception as e:
                    print(f"❌ Failed to load LoRA for {style}: {str(e)}")
                    raise
            
            with ThreadPoolExecutor() as executor:
                list(executor.map(load_lora, STYLE_CONFIGS.keys()))
            
            print("✨ Model load complete!")
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    def _inference(self, text: str, config: InferenceConfig):
        images = self.pipelines[config.style](
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            num_images_per_prompt=config.num_outputs
        ).images        
        return images

    def inference(self, text: str, config: InferenceConfig):
        print(f"🔦 Starting inference with prompt: {text}")
        images = self._inference(text, config)
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        print(f"🎉 Inference completed, got {len(image_output)} images!")
        return image_output
    
    @asgi_app(
        label="pixel-api",
    )
    def web_app(self):
        import os
        from fastapi import FastAPI, status, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.encoders import jsonable_encoder
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
        from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

        auth_scheme = HTTPBearer()
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
            allow_headers=["Content-Type", "Authorization", "prefer"],
        )
        
        def make_error_readable(error):
            return f"Field '{' -> '.join(error['loc'])}': {error['msg']}"
        
        @web_app.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            _, 
            exc: RequestValidationError
        ):
            readable_errors = [
                make_error_readable(error) for error in exc.errors()
            ]
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder({ "detail": readable_errors }),
            )
        
        @web_app.post("/v1/pixel")
        async def generate(body: BodyModel, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
            try:
                if token.credentials != os.environ["PIXEL_AUTH_TOKEN"]:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Unauthorized"},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                prompt = f"{STYLE_CONFIGS[body.input.style]['prompt_prefix']} {body.input.prompt}"
                
                print(f"🔦 Starting Pixel Color inference with prompt: {prompt}")
                
                images = self.inference(
                    text=prompt,
                    config=InferenceConfig(
                        num_outputs=body.input.num_outputs,
                        seed=body.input.seed,
                        style=body.input.style
                    ),
                )
                
                base64_images = [base64.b64encode(img).decode('utf-8') for img in images]
                print(f"🎉 Pixel Color inference completed, got {len(images)} images!")
                
                return JSONResponse(content={"images": base64_images})
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"detail": str(e)}
                )

        return web_app
    