from modal import App, Image as ModalImage, Volume, Secret, fastapi_endpoint, enter, Dict, parameter
from pydantic import BaseModel
from typing import TypedDict
from PIL import Image
from ut import generate_presigned_url, with_retry

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = ModalImage.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

image = (
    cuda_dev_image
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
        "torch",
        "torchao",
        "para-attn",
        "sqids",
        "numpy",
        "PyJWT",
        "cryptography"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_CACHE": "/cache",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    })
    .add_local_python_source("ut")
)

app = App(name="pixel-api")

inference_config = Dict.from_name("dotelier-inference-config")
styles = Dict.from_name("dotelier-styles")
http_config = Dict.from_name("dotelier-http-config")

class InputModel(BaseModel):
    prompt: str
    num_outputs: int = inference_config["NUM_OUTPUTS"]
    style: str = inference_config["DEFAULT_STYLE"]
    pixel_id: str

class InferenceConfig(InputModel):
    num_inference_steps: int = inference_config["NUM_INFERENCE_STEPS"]
    guidance_scale: float = inference_config["GUIDANCE_SCALE"]
    negative_prompt: str | None = None
    height: int = 1024
    width: int = 1024
    
class InferenceResult(TypedDict):
    images: list[Image.Image]
    inference_time: float

class JWKSCache:
    def __init__(self, jwks_url: str):
        self.jwks_url = jwks_url
        self.keys: Dict[str, dict] = {}
        
    def get_key(self, kid: str) -> dict:
        if kid not in self.keys:
            self.refresh_keys()
        return self.keys.get(kid)
    
    def refresh_keys(self):
        response = requests.get(self.jwks_url)
        jwks = response.json()
        self.keys = {key['kid']: key for key in jwks['keys']}

with image.imports():
    # FastAPI
    from fastapi import Response, HTTPException, status, Request, Depends
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    # ML/Processing imports
    import torch
    import pickle
    import gc
        
    # Uploads
    import math
    import hashlib
    import hmac
    from sqids import Sqids
    from sqids.constants import DEFAULT_ALPHABET
    from urllib.parse import urlencode
    import requests
    
    # Other utilities
    import os    
    import time
    import io
    import base64
    import jwt
    from jwt.algorithms import OKPAlgorithm
    from PIL import Image
 
@app.cls(
    gpu="H100",
    image=image,
    volumes={
        "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/quant": Volume.from_name("quant-cache", create_if_missing=True),
        "/images": Volume.from_name("dotelier-sample-images"),
    },
    secrets=[
        Secret.from_name("super-admin-user-id"),
        Secret.from_name("huggingface-secret"),
        Secret.from_name("pixel-auth-token"),
        Secret.from_name("uploadthing-secret"),
        Secret.from_name("uploadthing-app-id"),
    ],
    scaledown_window=1200,
    timeout=60 * 60,
)
class PixelModel:
    dev: str = parameter(default="false")
    styles = styles
    allowed_origins: list[str] = http_config["ALLOWED_ORIGINS"]
    jwks_cache = JWKSCache("https://dotelier.studio/api/auth/jwks")
        
    auth_scheme = HTTPBearer()
                 
    @enter()
    def load_model(self):
        self.dev = self.dev == "true"
        
        if self.dev:
            return
        
        try:
            print(f"loading model")
            
            from diffusers import FluxPipeline
            from torchao.quantization import autoquant
            from torchao.quantization.autoquant import AUTOQUANT_CACHE
            from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
            
            self.pipeline = FluxPipeline.from_pretrained(
                "graceyun/dotelier-color",
                torch_dtype=torch.bfloat16,
            ).to('cuda')
                        
            self.pipeline.transformer.fuse_qkv_projections()
            
            with open("/quant/vae_autoquant_cache.pkl", "rb") as f:
                AUTOQUANT_CACHE.update(pickle.load(f))
                
            self.pipeline.vae = autoquant(self.pipeline.vae, error_on_unseen=False)
            
            apply_cache_on_pipe(self.pipeline, residual_diff_threshold=0.12)
            
            torch.cuda.empty_cache()
            gc.collect()
            
            _ = self.pipeline(
                "a PXCON style icon, 16-bit",
                num_inference_steps=5,
                guidance_scale=5.0
            )
                    
            print("✨ model loaded and optimized")
        except Exception as e:
            print(f"[load_model] error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def authorize(self, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        if self.dev:
            return True
        return token.credentials == os.environ["PIXEL_AUTH_TOKEN"]
    
    def validate_origin(self, request: Request):
        if self.dev:
            return True
        origin = request.headers.get("Origin")
        return origin in self.allowed_origins
    
    def validate_jwt(self, token: str):
        try:   
            headers = jwt.get_unverified_header(token)
            kid = headers['kid']
            
            jwk = self.jwks_cache.get_key(kid)
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
        self, 
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
        
    def generate_prompt(self, token: str, prompt: str, suffix: str = ''):
        return f"a {token} style icon of: {prompt}{suffix}"
    
    @fastapi_endpoint(method="GET", docs=True, label="warmup")
    def warmup(
        self, 
        request: Request,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme)
    ):
        try:
            valid_origin = self.validate_origin(request.headers.get("Origin"))
            
            if not valid_origin:
                print(f"invalid origin: {request.headers.get('Origin')}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "forbidden"},
                )
                
            authorized = self.authorize(token)
            
            if not authorized:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "unauthorized"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return Response(
                content="success",
            )
        except Exception as e:
            print(f"[warmup] error: {str(e)}")
            return JSONResponse(
                    content={"detail": "unexpected error"},
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )               

    @fastapi_endpoint(method="POST", docs=True, label="generate")
    def generate(
        self,
        body: InputModel, 
        request: Request,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme)
    ):      
        try:
            valid_origin = self.validate_origin(request)
            
            if not valid_origin:
                print(f"invalid origin: {request.headers.get('Origin')}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Forbidden"},
                )
            
            authorized = self.authorize(token)
            
            if not authorized:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Unauthorized"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            auth_token = request.headers.get("X-Auth-JWT")
            user_id = self.validate_jwt(auth_token)
            
            if not user_id:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "invalid Supabase token"},
                )
                
            result = self.inference(body)
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
                        lambda: self.upload_thing(
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
        except Exception as e:
            print(f"[generate] error: {str(e)}")
            if isinstance(e, HTTPException):
                return Response(
                    content="unauthorized",
                    status_code=401,
                )
            else:
                return Response(
                    content="unexpected error",
                    status_code=500,
                )
             
    def inference(self, input: InputModel) -> InferenceResult:    
        if not self.dev and not self.pipeline:
            print("weird, pipeline was not loaded")
            self.load_model()
            
        style = input.style

        if not style in self.styles:
            raise ValueError(f"Unknown style: {style}")

        style_config = self.styles[style]
        token = style_config["token"]
        suffix = style_config["suffix"]
        negative_prompt = style_config["negative_prompt"]
        
        original_prompt = input.prompt
        prompt = self.generate_prompt(token, original_prompt, suffix)
        
        config = InferenceConfig(
            prompt=prompt,
            num_outputs=input.num_outputs,
            style=style,
            negative_prompt=negative_prompt,
            height=1024,
            width=1024,
            num_images_per_prompt=input.num_outputs,
            pixel_id=input.pixel_id,
        )
        
        if self.dev:
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
                negative_prompt=config.negative_prompt,
                height=config.height,
                width=config.width,
            ).images
            
        elapsed = time.time() - start_time
        print(f"completed in {elapsed:.2f} seconds")

        return InferenceResult(
            images=images,
            inference_time=elapsed,
        )
