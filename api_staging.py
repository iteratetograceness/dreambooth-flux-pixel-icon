"""STAGING copy of api.py serving the v2 fine-tune candidate.

Differences from api.py (prod), on purpose:
  - App name / asgi label: dotelier-api-staging
  - LORA_REPO: graceyun/dotelier-pixel-v2-ckpt750 (winning checkpoint of the
    lr_0.0001_steps_1500_rank_16_bs_4 run — see eval-results/SUMMARY.md)
  - num_inference_steps 28 (was 50), guidance_scale 3.5 (was 7.5) — the
    training-matched settings the eval sheets validated
  - warmup uses the training-matched template at guidance 3.5
  - cost: buffer_containers=0 and scaledown_window=300 — staging shouldn't
    keep a warm B200 around; restore prod's values on promotion
  - smoke(): `modal run api_staging.py` generates one image through the real
    container as a deploy smoke test

Promotion = copying LORA_REPO + the config dict values back into api.py.
"""
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

# python 3.12: the pinned stack (numpy 2.5.1) requires >=3.12, and the eval
# runs that validated these pins ran on 3.12
cuda_dev_image = ModalImage.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.12"
).entrypoint([])

# Pre-download model weights at image build time so they're baked into the
# container image layer. This avoids ~22GB HuggingFace downloads on cold start.
MODEL_DIR = "/model"
BASE_MODEL = "black-forest-labs/FLUX.1-dev"
# v2.1 winner: dataset v2.1 (crisp, 62% fill), lr 1e-4, checkpoint 1000.
# Template + guidance below are part of the recipe (template shootout,
# eval-results/09_template_shootout.png).
LORA_REPO = "graceyun/dotelier-pixel-v21-ckpt1000"
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"


def download_models():
    from huggingface_hub import snapshot_download, hf_hub_download

    # Skip the single-file BFL-format checkpoints (~24GB) that the
    # diffusers-layout FluxPipeline never reads — roughly 40% of the repo.
    # TODO: pin revision= for reproducible rebuilds.
    snapshot_download(
        BASE_MODEL,
        local_dir=f"{MODEL_DIR}/base",
        ignore_patterns=[
            "flux1-dev.safetensors",
            "ae.safetensors",
            "*.md",
            "*.png",
            ".gitattributes",
        ],
    )
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
        # Stack refreshed 2026-07, all exact-pinned. Validated end-to-end by
        # the eval runs, which resolved these same versions and successfully
        # loaded FLUX + LoRA and generated. B200 (Blackwell, sm_100) needs
        # sm_100 kernels: torch 2.13's default wheel bundles CUDA 13, which
        # requires host driver r580+ (Modal B200 hosts run 580.x — verified).
        # If a rebuild ever lands on older drivers, fall back to torch==2.9.1
        # (cu128, also sm_100-capable). Bump pins deliberately, never float.
        "diffusers==0.39.0",
        "transformers==5.14.0",
        "accelerate==1.14.0",
        "safetensors==0.8.0",
        "peft==0.19.1",
        "huggingface_hub==1.23.0",
        "fastapi[standard]",
        "pydantic>=2.0",
        "sentencepiece==0.2.2",
        "torch==2.13.0",
        "numpy==2.5.1",
    )
    .env({
        # hub 1.x replaced hf_transfer with built-in Xet transfer
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTORCH_ALLOC_CONF": "max_split_size_mb:512",
    })
    .run_function(
        download_models,
        secrets=[Secret.from_name("huggingface-secret")],
    )
    .add_local_python_source("ut")
)

app = App(name="dotelier-api-staging")

config = {
   "num_outputs": 1,
   "num_inference_steps": 28,
   "guidance_scale": 5.0,
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
    # the composed prompt (template + user prompt) can exceed the user-facing
    # 500-char cap — don't re-validate it here or 465+ char prompts 500
    prompt: str
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
        "/images": Volume.from_name("dotelier-sample-images"),
    },
    secrets=[
        Secret.from_name("huggingface-secret"),
        Secret.from_name("vercel-automation-bypass"),
    ],
    max_containers=5,
    min_containers=0,
    buffer_containers=0,   # staging: no warm spare B200 (prod uses 1)
    scaledown_window=300,  # staging: 5 min idle (prod uses 30 min)
    timeout=60 * 60,
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

            # Load from baked-in local paths (no network download on cold start).
            self.pipeline = FluxPipeline.from_pretrained(
                f"{MODEL_DIR}/base",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to('cuda')

            self.pipeline.load_lora_weights(
                f"{MODEL_DIR}/lora",
                weight_name=LORA_WEIGHTS,
            )

            # fuse once at startup: numerically equivalent output, but removes
            # the per-projection lora_A/lora_B matmul overhead that otherwise
            # runs on every step of every request
            self.pipeline.fuse_lora()
            self.pipeline.unload_lora_weights()

            _ = self.pipeline(
                self.generate_prompt("a floppy disk"),
                num_inference_steps=5,
                guidance_scale=config["guidance_scale"],
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
        return (
            f"a PXCON, a simple chunky 8-bit pixel art icon of {prompt}, "
            "thick black outline, flat colors, on a plain white background"
        )

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
        # super-admin-user-id and huggingface-secret removed: nothing in the
        # web path reads them, and the HF token is write-capable — an
        # internet-facing container shouldn't carry it
        Secret.from_name("pixel-auth-token"),
        Secret.from_name("uploadthing-secret"),
        Secret.from_name("uploadthing-app-id"),
        Secret.from_name("vercel-automation-bypass"),
    ],
    max_containers=10,
    min_containers=0,
    buffer_containers=0,  # staging: prod keeps 2
    scaledown_window=300,
    # Modal's default 300s is shorter than a worst-case B200 scale-from-zero
    # (weight streaming + warmup + generate + uploads); stopgap until
    # /generate moves to spawn + poll
    timeout=900,
)
@asgi_app(label='dotelier-api-staging')
def fastapi_app():
    from fastapi import FastAPI, Response, HTTPException, status, Request, Depends
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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


@app.local_entrypoint()
def smoke(prompt: str = "a cactus in a terracotta pot"):
    """Deploy smoke test: run one real generation through the staging
    container (same image/GPU/config as the endpoint) and save it locally."""
    result = PixelModel().inference.remote(
        InputModel(prompt=prompt, pixel_id="smoke-test")
    )
    img = result["images"][0]
    out = "staging_smoke.png"
    img.save(out)
    print(f"generated in {result['inference_time']:.2f}s -> {out}")
