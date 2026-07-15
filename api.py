from modal import App, Image as ModalImage, Volume, Secret, enter, Dict, parameter, asgi_app, method
from pydantic import BaseModel, Field
from typing import TypedDict
from PIL import Image
import requests
import os

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = ModalImage.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

# Pre-download model weights at image build time so they're baked into the
# container image layer. This avoids ~22GB HuggingFace downloads on cold start.
MODEL_DIR = "/model"
BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_REPO = "graceyun/lr_0.0002_steps_4200_rank_16_031325"
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"


def download_models():
    import os
    # Disable hf_transfer for build step — it's a C extension that may not
    # be available in the build container. Regular downloads are fine here
    # since this only runs once at image build time.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    from huggingface_hub import snapshot_download, hf_hub_download

    snapshot_download(BASE_MODEL, local_dir=f"{MODEL_DIR}/base")
    hf_hub_download(
        LORA_REPO,
        filename=LORA_WEIGHTS,
        local_dir=f"{MODEL_DIR}/lora",
    )


image = (
    cuda_dev_image
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers==0.31.0",
        "transformers==4.44.0",
        "accelerate==0.34.0",
        "safetensors==0.4.5",
        "peft==0.13.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "fastapi[standard]",
        "pydantic>=2.0",
        "sentencepiece==0.2.0",
        "torch>=2.6.0",
        # "torchao",
        # "para-attn",
        "numpy<2",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Removed HF_HUB_CACHE — weights are now baked into the image at MODEL_DIR,
        # so we no longer need a runtime cache volume for downloads.
        "PYTORCH_ALLOC_CONF": "max_split_size_mb:512",
    })
    .run_function(
        download_models,
        secrets=[Secret.from_name("huggingface-secret")],
    )
    .add_local_python_source("ut")
)

app = App(name="dotelier-api")

config = {
   "num_outputs": 1,
   "num_inference_steps": 50,
   "guidance_scale": 7.5,
   "default_style": "color",
   "allowed_origins": [
    "https://dotelier.studio",
    "https://www.dotelier.studio",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
   ],
   "styles": {
       "color": {
            "token": "PXCON",
            "suffix": ", pixel art, 16-bit style, clean minimal design, white background, sharp uniform black outline, no anti-aliasing, no blur, hard edges",
            "negative_prompt": "ugly, blurry, noisy, messy, dirty, complex, detailed, photorealistic, 3d, gradient, shading, multiple outlines, double outline, text that says pxcon, text that says PXCON, written text, signature, watermark, low quality, distorted, deformed",
        }
   }
}

class InputModel(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    num_outputs: int = Field(default=config["num_outputs"], ge=1, le=4)
    pixel_id: str = Field(min_length=1, max_length=128)
    style: str = config["default_style"]

class InferenceConfig(InputModel):
    num_inference_steps: int = config["num_inference_steps"]
    guidance_scale: float = config["guidance_scale"]
    negative_prompt: str | None = None
    height: int = 1024
    width: int = 1024
    
class InferenceResult(TypedDict):
    images: list[Image.Image]
    inference_time: float

class JWKSCache:
    # re-fetch periodically so rotated/revoked signing keys stop validating
    # even on long-lived warm containers
    TTL_SECONDS = 15 * 60

    def __init__(self, jwks_url: str):
        self.jwks_url = jwks_url
        self.keys: Dict[str, dict] = {}
        self.fetched_at: float = 0.0

    def get_key(self, kid: str) -> dict:
        import time as _time
        if kid not in self.keys or _time.monotonic() - self.fetched_at > self.TTL_SECONDS:
            self.refresh_keys()
        return self.keys.get(kid)
    
    def refresh_keys(self):
        response = requests.get(
            self.jwks_url,
            headers={"x-vercel-protection-bypass": os.environ["VERCEL_AUTOMATION_BYPASS_SECRET"]},
            timeout=10,
        )
        response.raise_for_status()
        jwks = response.json()
        self.keys = {key['kid']: key for key in jwks['keys']}
        import time as _time
        self.fetched_at = _time.monotonic()

with image.imports():
    import torch
    import pickle
    import gc
    import time

@app.cls(
    gpu="B200",
    image=image,
    volumes={
        # "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),  # No longer needed — weights baked into image
        # "/quant": Volume.from_name("quant-cache", create_if_missing=True),
        "/images": Volume.from_name("dotelier-sample-images"),
    },
    secrets=[
        Secret.from_name("huggingface-secret"),
        Secret.from_name("vercel-automation-bypass"),
    ],
    max_containers=5,
    min_containers=0,
    buffer_containers=1,
    scaledown_window=1800,  # 30 min idle before scaledown (was 20 min)
    timeout=60 * 60,
    # enable_memory_snapshot=True, - Causing weird timing issues
    # experimental_options={"enable_gpu_snapshot": True}, - Not working
)
class PixelModel:
    dev: str = parameter(default="false")
    styles = config["styles"]
    is_dev: bool = False
                 
    @enter()
    def load_model(self):
        self.is_dev = self.dev == "true"
        
        if self.is_dev:
            return
        
        try:
            print(f"loading model")
            
            from diffusers import FluxPipeline
            # from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

            # Load from baked-in local paths (no network download on cold start).
            # To revert to runtime download, swap these back and re-enable /cache volume.
            self.pipeline = FluxPipeline.from_pretrained(
                f"{MODEL_DIR}/base",  # was: "black-forest-labs/FLUX.1-dev"
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to('cuda')

            self.pipeline.load_lora_weights(
                f"{MODEL_DIR}/lora",  # was: "graceyun/lr_0.0002_steps_4200_rank_16_031325"
                weight_name=LORA_WEIGHTS,  # was: "pytorch_lora_weights.safetensors"
            )

            # self.pipeline.transformer.fuse_qkv_projections()
            # apply_cache_on_pipe(self.pipeline, residual_diff_threshold=0.06)

            _ = self.pipeline(
                "a PXCON style icon, 16-bit",
                num_inference_steps=5,
                guidance_scale=5.0
            )
            
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
                    
            print("✨ model loaded and optimized")
        except Exception as e:
            print(f"[load_model] error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def generate_prompt(self, prompt: str):
        return f"a PXCON, a 16-bit pixel art icon of {prompt}"
     
    @method()
    def inference(self, input: InputModel) -> InferenceResult:
        if not self.is_dev and getattr(self, "pipeline", None) is None:
            print("weird, pipeline was not loaded")
            self.load_model()

        style = input.style

        if not style in self.styles:
            raise ValueError(f"Unknown style: {style}")

        original_prompt = input.prompt
        prompt = self.generate_prompt(original_prompt)

        config = InferenceConfig(
            prompt=prompt,
            num_outputs=input.num_outputs,
            style=style,
            height=1024,
            width=1024,
            pixel_id=input.pixel_id,
        )
        
        if self.is_dev:
            print(f"dev mode, returning test image")
            image = Image.open("/images/test.png")
            return InferenceResult(
                    images=[image],
                    inference_time=4.44,
                )
            
        print(f"starting inference with prompt: {original_prompt}")
        start_time = time.time()
        
        with torch.inference_mode():
            images = self.pipeline(
                prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                num_images_per_prompt=config.num_outputs,
                height=config.height,
                width=config.width,
            ).images
            
        elapsed = time.time() - start_time
        print(f"completed in {elapsed:.2f} seconds")

        return InferenceResult(
            images=images,
            inference_time=elapsed,
        )

web_image = ModalImage.debian_slim().pip_install(
    "fastapi[standard]", 
    "sqids", 
    "PyJWT", 
    "cryptography", 
    "Pillow", 
    "requests"
    ).add_local_python_source("ut")

@app.function(
    image=web_image,
    secrets=[
        Secret.from_name("super-admin-user-id"),
        Secret.from_name("huggingface-secret"),
        Secret.from_name("pixel-auth-token"),
        Secret.from_name("uploadthing-secret"),
        Secret.from_name("uploadthing-app-id"),
        Secret.from_name("vercel-automation-bypass"),
    ],
    max_containers=10,
    min_containers=0,
    buffer_containers=2,
    scaledown_window=300,
)
@asgi_app(label='dotelier-api')
def fastapi_app():
    from fastapi import FastAPI, Response, HTTPException, status, Request, Depends
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    # import math
    # import hashlib
    # import hmac
    # from sqids import Sqids
    # from sqids.constants import DEFAULT_ALPHABET
    # from urllib.parse import urlencode
    import hmac
    import io
    import os
    import requests
    import base64
    import jwt
    from jwt.algorithms import OKPAlgorithm
    from ut import generate_presigned_url, with_retry
    
    web_app = FastAPI()
    
    allowed_origins: list[str] = config["allowed_origins"]
    jwks_cache = JWKSCache("https://dotelier.studio/api/auth/jwks")
        
    auth_scheme = HTTPBearer()
    
    def authorize(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        return hmac.compare_digest(token.credentials, os.environ["PIXEL_AUTH_TOKEN"])
    
    def validate_origin(request: Request):
        origin = request.headers.get("Origin")
        return origin in allowed_origins
    
    def validate_jwt(token: str):
        try:   
            headers = jwt.get_unverified_header(token)
            kid = headers['kid']
            
            jwk = jwks_cache.get_key(kid)
            if not jwk:
                raise ValueError("no matching key found")   
            
            algorithm = OKPAlgorithm()
            key = algorithm.from_jwk(jwk)
            
            payload = jwt.decode(
                token,
                key,
                algorithms=["EdDSA"],
                options={
                    "verify_aud": False,
                }
            )
            
            user_id = payload.get("id")
            if not user_id:
                raise ValueError("user id not found in token")
            
            return user_id
        except Exception as e:
            print(f"[validate_jwt] error: {str(e)}")
            return None
        
    def upload_thing(
        file_name: str, 
        file_size: int, 
        file_type: str = "image/png", 
        file_bytes: bytes = None
        ):        
        presigned_data = generate_presigned_url(
            app_id=os.environ["UPLOADTHING_APP_ID"],
            api_key=os.environ["UPLOADTHING_API_KEY"],
            file_name=file_name,
            file_size=file_size,
            file_type=file_type,
        )
        
        response = requests.put(
            presigned_data["url"],
            files={"file": file_bytes},
            timeout=60,
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to upload icon: {response.text}")
        
        response_body = response.json()
        file_key = presigned_data["file_key"]
        file_url = response_body["ufsUrl"]
        
        return {
            "success": True,
            "url": file_url,
            "file_key": file_key
        }

    # @web_app.get("/warm")
    # def warm(
    #     request: Request,
    #     token: HTTPAuthorizationCredentials = Depends(auth_scheme)
    # ):
    #     try:
    #         valid_origin = validate_origin(request)
            
    #         if not valid_origin:
    #             print(f"invalid origin: {request.headers.get('Origin')}")
    #             return JSONResponse(
    #                 status_code=status.HTTP_403_FORBIDDEN,
    #                 content={"detail": "forbidden"},
    #             )
                
    #         authorized = authorize(token)
            
    #         if not authorized:
    #             return JSONResponse(
    #                 status_code=status.HTTP_401_UNAUTHORIZED,
    #                 content={"detail": "unauthorized"},
    #                 headers={"WWW-Authenticate": "Bearer"},
    #             )
            
    #         return Response(
    #             content="success",
    #         )
    #     except Exception as e:
    #         print(f"[warmup] error: {str(e)}")
    #         return JSONResponse(
    #                 content={"detail": "unexpected error"},
    #                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #             )               

    @web_app.post("/generate")
    def generate(
        body: InputModel, 
        request: Request,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme)
    ):      
        try:
            valid_origin = validate_origin(request)
            
            if not valid_origin:
                print(f"invalid origin: {request.headers.get('Origin')}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Forbidden"},
                )
            
            authorized = authorize(token)
            
            if not authorized:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Unauthorized"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            auth_token = request.headers.get("X-Auth-JWT")
            user_id = validate_jwt(auth_token)
            
            if not user_id:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "invalid jwt token"},
                )
            
            model = PixelModel()
            result = model.inference.remote(body)
            images = result["images"]
            elapsed = result["inference_time"]
            
            processed_images = []
            
            for image in images:
                with io.BytesIO() as buf:
                    image.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    
                    file_name = f"{body.pixel_id}.png"
                    
                    upload_result = with_retry(
                        lambda: upload_thing(
                            file_name=file_name,
                            file_size=len(image_bytes),
                            file_type="image/png",
                            file_bytes=image_bytes
                        )
                    )
                    
                    processed_images.append({
                        "base64": base64_image,
                        "url": upload_result["url"],
                        "fileKey": upload_result["file_key"]
                    })
                    
            return JSONResponse(
                content={"images": processed_images, "inference_time": elapsed},
            )
        except HTTPException:
            raise
        except Exception as e:
            print(f"[generate] error: {str(e)}")
            return Response(
                content="unexpected error",
                status_code=500,
            )
     

    return web_app