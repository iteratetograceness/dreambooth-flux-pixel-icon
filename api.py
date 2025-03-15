from modal import App, Image as ModalImage, Volume, Secret, gpu, asgi_app, enter, method
from pydantic import BaseModel
from enum import Enum

NUM_INFERENCE_STEPS = 60
GUIDANCE_SCALE = 5
NUM_OUTPUTS = 1
SUFFIX = ", 16-bit, on white background, only use up to 7 colors inclusive of #fff and #000, only use #fff for the background and #000 for black outline"

def generate_prompt(token: str, prompt: str):
    return f"a {token} style icon of: {prompt}{SUFFIX}"

def extract_user_prompt(prompt: str):
    return prompt.split("of:")[1].strip().replace(SUFFIX, '')

STYLE_CONFIGS = {
    "color_v2": {
        "dir": "graceyun/lr_0.0002_steps_4200_rank_16_031325",
        "token": "PXCON",
        "negative_prompt": 'text:"PXCON", text:"pxcon", logo',
    }
}
DEFAULT_STYLE = "color_v2"

huggingface_secret = Secret.from_name("huggingface-secret")
auth_token = Secret.from_name("pixel-auth-token")
supabase_url = Secret.from_name("supabase-url")
supabase_service_role_key = Secret.from_name("supabase-service-role-key")
supabase_jwt_secret = Secret.from_name("supabase-jwt-secret")
recraft_api_key = Secret.from_name("recraft-api-key")
bg_remover_model = Secret.from_name("bg-remover-model")
replicate_api_token = Secret.from_name("replicate-api-token")

class InputModel(BaseModel):
    prompt: str
    num_outputs: int = 1
    seed: int | None = None
    style: str = DEFAULT_STYLE
class BodyModel(BaseModel):
    input: InputModel
class InferenceConfig(BaseModel):
    style: str = DEFAULT_STYLE
    num_inference_steps: int = NUM_INFERENCE_STEPS
    guidance_scale: float = GUIDANCE_SCALE
    num_outputs: int = 1
    seed: int | None = None
    negative_prompt: str | None = None
class JobStatus(str, Enum):
    QUEUED = 'queued'
    INITIATED = 'initiated'
    INFERENCE = 'inference'
    POST_PROCESSING = 'post_processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

api_image = (
    ModalImage.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi[standard]",
        "pydantic>=2.0",
        "supabase",
        "pyjwt[crypto]",
    )
)

pp_image = (
    ModalImage.debian_slim(python_version="3.10")
    .pip_install(
        "requests",
        "replicate",
        "pillow",
    )
)

image = (
    ModalImage.debian_slim(python_version="3.10")
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
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_CACHE": "/cache",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    })
)

app = App(name="pixel-api")

@app.function(
    image=api_image,
    secrets=[auth_token, supabase_url, supabase_service_role_key, supabase_jwt_secret],
    scaledown_window=300,
)
@asgi_app(label="pixel-api")
def web_app():
    import os
    from fastapi import FastAPI, status, Depends, Request, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    auth_scheme = HTTPBearer()
    base_url = web_app.web_url
    api = FastAPI(title="Pixel API", description="Generate pixel icons with job queue")
    
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://dotelier.studio",
            "https://www.dotelier.studio",
        ],
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "prefer"],
    )
    
    def make_error_readable(error):
        return f"Field '{' -> '.join(error['loc'])}': {error['msg']}"
    
    @api.exception_handler(RequestValidationError)
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
    
    @api.get("/warmup")
    async def warmup():
        try:
            background_tasks = BackgroundTasks()
            background_tasks.add_task(warm_up_model)
            
            return JSONResponse(
                content={"success": True},
                background=background_tasks
            )
        except Exception as e:
            print(f"Error initiating warm-up: {str(e)}")
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500
            )

    async def warm_up_model():
        try:
            from api import process_pixel_job
            
            process_pixel_job.spawn(
                "warm-up prompt", 
                {
                    "style": "color_v2",
                    "num_inference_steps": 1,
                    "num_outputs": 1,
                    "is_warmup": True
                }
            )
            
            print("✅ Warm-up job submitted")
        except Exception as e:
            print(f"❌ Error in warm-up process: {str(e)}")

    def validate_supabase_jwt(token):
        try:
            import jwt
        
            payload = jwt.decode(
                token,
                os.environ["SUPABASE_JWT_SECRET"],
                algorithms=["HS256"],
                options={"verify_aud": False}
            )
            
            user_id = payload.get("sub")
            if not user_id:
                raise ValueError("User ID not found in token")
            
            return user_id
        except Exception as e:
            print(f"❌ Error validating Supabase JWT: {str(e)}")
            return None
            
    @api.post("/v1/pixel")
    async def generate(body: BodyModel, request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        from api import process_pixel_job
        from supabase import create_client
            
        try:
            if token.credentials != os.environ["PIXEL_AUTH_TOKEN"]:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Unauthorized"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            supabase_token = request.headers.get("X-Supabase-Auth")
            if not supabase_token:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Supabase authentication required"},
                )
            
            user_id = validate_supabase_jwt(supabase_token)
            if not user_id:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid Supabase token"},
                )
            
            style = body.input.style
            style_config = STYLE_CONFIGS[style]
            token = style_config['token']
            negative_prompt = style_config['negative_prompt']
            user_prompt = body.input.prompt
            prompt = generate_prompt(token, user_prompt)
            
            print(f"🔦 Submitting job for prompt: {prompt}")
            
            supabase = create_client(
                supabase_url=os.environ["SUPABASE_URL"],
                supabase_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"]
            )
             
            result = supabase.table("jobs").insert({
                "user_id": user_id,
                "prompt": prompt,
                "style": style,
                "status": JobStatus.QUEUED
            }).execute()
            
            job_id = result.data[0]['id']
            
            callback_url = f"{base_url}/v1/pixel/callback"
            
            import random
            seed = random.randint(0, 2**32 - 1)

            config = {
                "style": body.input.style,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "num_outputs": NUM_OUTPUTS,
                "seed": seed,
                "negative_prompt": negative_prompt,
            }
            
            metadata = {
                "job_id": job_id,
                "user_id": user_id,
            }
            
            call = process_pixel_job.spawn(prompt, config, callback_url, metadata)
            modal_job_id = call.object_id
            
            supabase.table("jobs").update({
                "modal_job_id": modal_job_id
            }).eq("id", job_id).execute()
            
            print(f"✅ Job submitted with ID: {job_id}")
            
            return JSONResponse(
                content={
                    "job_id": job_id,
                    "status": JobStatus.QUEUED
                }
            )
        except Exception as e:
            print(f"Error during job submission: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)}
            )
        
    @api.post("/v1/pixel/callback")
    async def callback(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        try:
            if token.credentials != os.environ["PIXEL_AUTH_TOKEN"]:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Unauthorized"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            data = await request.json()
            job_id = data.get("job_id")
            status = data.get("status")
            message = data.get("message")
            images = data.get("images", [])
            error = data.get("error")
            user_id = data.get("user_id")
            prompt = data.get("prompt")
            
            print(f"📩 Received callback for job {job_id}: {status} - {message}")
            
            from supabase import create_client
            import datetime
            
            supabase = create_client(
                supabase_url=os.environ["SUPABASE_URL"],
                supabase_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"]
            )
            
            update_data = {
                "status": status,
                "message": message,
                "updated_at": datetime.datetime.now().isoformat()
            }
            
            if error:
                update_data["error_message"] = error.get("message")
                update_data["error_stage"] = error.get("stage")
            
            supabase.table("jobs").update(update_data).eq("id", job_id).execute()
                        
            if status == JobStatus.COMPLETED and images and len(images) > 0:
                background_tasks = BackgroundTasks()
                background_tasks.add_task(
                    upload_svg, 
                    supabase,
                    job_id, 
                    images[0], # MVP only handles a single image; refactor to handle multiple
                    user_id,
                    prompt
                )
                
                return JSONResponse(content={"success": True}, background=background_tasks)
            
            return JSONResponse(content={"success": True})
        except Exception as e:
            print(f"Error processing callback: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)}
            )
    
    async def upload_svg(supabase, job_id, image_data, user_id, prompt):        
        import base64
        
        try:
            svg_b64 = image_data.get("formats", {}).get("svg")
            
            if not svg_b64:
                print("❌ No SVG found in image data")
                return

            storage_path = f"{job_id}.svg"
            svg_bytes = base64.b64decode(svg_b64)
            storage_result = supabase.storage.from_("icons").upload(
                storage_path,
                svg_bytes,
                {"content-type": "image/svg+xml"}
            )
            
            result_path = storage_result.path

            supabase.table("jobs").update({
                "result_url": result_path
            }).eq("id", job_id).execute()
            
            supabase.table("pixel").insert({
                "prompt": extract_user_prompt(prompt),
                "job_id": job_id,
                "user_id": user_id,
                "file_path": result_path
            }).execute()
            
            print(f"✅ Successfully uploaded SVG for job {job_id}")            
        except Exception as e:
            print(f"❌ Error uploading SVG: {str(e)}")
            
    @api.get("/v1/pixel/job/{job_id}")
    async def get_job_result(job_id: str, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        try:
            if token.credentials != os.environ["PIXEL_AUTH_TOKEN"]:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Unauthorized"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            print(f"🔍 Checking status for job: {job_id}")
            
            from supabase import create_client
            
            supabase = create_client(
                supabase_url=os.environ["SUPABASE_URL"],
                supabase_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"]
            )
            
            response = supabase.table("jobs").select("*").eq("id", job_id).single().execute()
            
            if not response.data:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Job not found: {job_id}"}
                )
            
            job = response.data
            
            if job["status"] == JobStatus.COMPLETED:
                return JSONResponse(
                    content={
                        "job_id": job_id,
                        "status": job["status"],
                        "result_url": job["result_url"],
                        "message": job["message"]
                    }
                )
            
            if job["status"] == JobStatus.FAILED:
                return JSONResponse(
                    content={
                        "job_id": job_id,
                        "status": job["status"],
                        "error": {
                            "message": job["error_message"],
                            "stage": job["error_stage"]
                        },
                        "message": job["message"]
                    }
                )
                
            return JSONResponse(
                status_code=202,
                content={
                    "job_id": job_id,
                    "status": job["status"],
                    "message": job["message"]
                }
            )            
        except Exception as e:
            print(f"Error checking job status: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)}
            )
    
    return api

@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
    secrets=[huggingface_secret, auth_token, recraft_api_key, bg_remover_model, replicate_api_token],
    timeout=10 * 60,
)
def process_pixel_job(prompt: str, config: dict, callback_url: str = None, metadata: dict = None):
    model = PixelModel()
    return model.process_job_queue.remote(prompt, config, callback_url, metadata)

@app.function(
    image=pp_image,
    timeout=5 * 60,
    secrets=[recraft_api_key, bg_remover_model, replicate_api_token],
)
def post_process_image(image_data: bytes):
    def remove_background(image_data: str) -> bytes:
        import replicate
        import requests
        import os
        
        data_uri = f"data:image/png;base64,{image_data}"
        
        input = {
            "image": data_uri,
        }
        
        model = os.environ["BG_REMOVER_MODEL"]
                
        output = replicate.run(
            model,
            input=input
        )
        
        response = requests.get(output)
        
        if response.status_code != 200:
            raise Exception(f"Failed to remove background: {response.status_code}, {response.text}")
        
        return response.content
        
    def convert_to_svg(image_data: bytes) -> dict:
        import requests
        from PIL import Image
        import io
        
        image_buffer = io.BytesIO(image_data)
        image = Image.open(image_buffer)
        image = image.resize((image.width * 2, image.height * 2))
        
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        files = {
            'file': ('image.png', image_data, 'image/png')
        }
        
        data = {
            'response_format': 'b64_json'
        }

        headers = {}
        headers['Authorization'] = 'Bearer 6xdz0qUnWjwX262od2ZZru65iBkRnvnWHu8CG0Y8CPgBvSWWb7yXNgUP4wUN7hc1'
        
        response = requests.post(
            "https://external.api.recraft.ai/v1/images/vectorize",
            data=data,
            files=files,
            headers=headers,
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to convert to SVG: {response.status_code}, {response.text}")
        
        data = response.json()
                
        if not data or not data.get('image') or not data['image'].get('b64_json'):
            raise Exception(f"Failed to convert to SVG: {data}")
        
        return data['image']['b64_json']

    try:
        removed_background = remove_background(image_data)
        vectorize_result = convert_to_svg(removed_background)
        
        return vectorize_result
    except Exception as e:
        print(f"Error in post-processing: {str(e)}")
        raise

@app.cls(
    gpu="H100",
    image=image,
    volumes={
        "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
    secrets=[huggingface_secret, auth_token, supabase_url, supabase_service_role_key, recraft_api_key, bg_remover_model, replicate_api_token],
    scaledown_window=5 * 60,
    timeout=10 * 60,
)
class PixelModel:
    def __init__(self) -> None:
        self.model_loaded = False
        self.pipelines = {}
        self.recraft_api_url = "https://external.api.recraft.ai/v1/images/vectorize"
        self.model_loading_initiated = False
                 
    @enter()
    def load_model(self):
        self.start_model_loading_thread()
    
    def start_model_loading_thread(self):
        import threading
        
        if self.model_loading_initiated:
            return
        
        self.model_loading_initiated = True
        
        def load_models_thread():
            try:
                import torch 
                from diffusers import FluxPipeline
                
                print(f"🤗 Starting model load")
                                        
                torch.backends.cudnn.benchmark = True  

                for style, config in STYLE_CONFIGS.items():
                    try:
                        print(f"🎯 Loading model for {style}...")
                        
                        # Load model directly from graceyun/dotelier-color which has LoRA weights fused
                        # This eliminates the need to load the base model and LoRA weights separately
                        pipeline = FluxPipeline.from_pretrained(
                                "graceyun/dotelier-color",
                                torch_dtype=torch.bfloat16,
                        )

                        pipeline.vae.enable_slicing()
                        pipeline.vae.enable_tiling()
                        pipeline.to('cuda')

                        print("✅ Fused model loaded from graceyun/dotelier-color")

                        # Enable torch.compile for faster inference
                        print("⚡ Compiling transformer for faster inference...")
                        try:
                            pipeline.transformer = torch.compile(
                                pipeline.transformer,
                                mode="max-autotune",
                                fullgraph=True
                            )
                            print("✅ Transformer compilation complete")
                        except Exception as e:
                            print(f"⚠️ Compilation failed, falling back to uncompiled: {e}")
                        
                        self.pipelines[style] = pipeline
                        print(f"✨ LoRA weights loaded + compiled for {style}!")
                    except Exception as e:
                        print(f"❌ Failed to load LoRA for {style}: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        raise
                
                self.model_loaded = True
                print("✨ Model(s) loaded and optimized successfully!")
            except Exception as e:
                print(f"❌ Failed to load model: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise
        
        threading.Thread(target=load_models_thread).start()
    
    def ensure_models_loaded(self):
        import time
        
        if not self.model_loading_initiated:
            self.start_model_loading_thread()
        
        if not self.model_loaded:
            print("⏳ Waiting for models to finish loading...")
            max_wait_seconds = 300
            wait_interval = 10
            total_waited = 0
            
            while not self.model_loaded and total_waited < max_wait_seconds:
                time.sleep(wait_interval)
                total_waited += wait_interval
                print(f"⏳ Still waiting for models... ({total_waited}s)")
            
            if not self.model_loaded:
                raise Exception(f"Timed out waiting for models to load after {max_wait_seconds} seconds")
            
            print("✅ Models finished loading and are ready to use")
            
    def update_job_status(self, callback_url, status, metadata, message=None, images=None, error_details=None):
        import os
        import requests
        
        if not callback_url or not metadata:
            return
            
        job_id = metadata.get("job_id")
        user_id = metadata.get("user_id")
        prompt = metadata.get("prompt")
        
        payload = {
            "job_id": job_id,
            "user_id": user_id,
            "prompt": prompt,
            "status": status
        }
        
        if message:
            payload["message"] = message
            
        if images:
            payload["images"] = images
            
        if error_details:
            payload["error"] = {
                "message": str(error_details.get("message", "")),
                "stage": error_details.get("stage", "unknown"),
                "traceback": error_details.get("traceback", ""),
                "timestamp": error_details.get("timestamp", "")
            }
            
        try:
            response = requests.post(
                callback_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ['PIXEL_AUTH_TOKEN']}"
                }
            )
            if response.status_code != 200:
                print(f"Warning: Status update returned non-200 response: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Warning: Failed to send status update: {e}")
             
    def inference(self, text: str, config: InferenceConfig):
        import time
        
        self.ensure_models_loaded()
        
        start_time = time.time()
        print(f"🔦 Starting inference with prompt: {text}")

        images = self.pipelines[config.style](
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            num_images_per_prompt=config.num_outputs,
            negative_prompt=config.negative_prompt
        ).images        
        
        elapsed = time.time() - start_time
        print(f"⚡ Generation completed in {elapsed:.2f} seconds")

        return images
                
    def process_job(self, text: str, config: InferenceConfig, job_id: str, callback_url: str, user_id: str):
        import time
        import io
        import base64
        
        metadata = {
            "prompt": text,
            "job_id": job_id,
            "user_id": user_id
        }
        
        try:
            self.update_job_status(callback_url, JobStatus.INITIATED, metadata, "Starting job and preparing model")
            
            self.ensure_models_loaded()
            
            start_time = time.time()
            
            # STAGE 1: INFERENCE
            self.update_job_status(callback_url, JobStatus.INFERENCE, metadata, "Starting inference")
            images = self.inference(text, config)
            print(f"✅ Generated {len(images)} images")
            
            processed_images = []
            for idx, image in enumerate(images):
                result = {
                    "index": idx,
                    "formats": {}
                }
                                
                with io.BytesIO() as buf:
                    image.save(buf, format="PNG")
                    original_bytes = buf.getvalue()
                    result["formats"]["original"] = base64.b64encode(original_bytes).decode('utf-8')
                
                # STAGE 2: POST-PROCESSING
                try:
                    self.update_job_status(
                        callback_url, 
                        JobStatus.POST_PROCESSING, 
                        metadata,
                        f"Post-processing image {idx+1}/{len(images)}"
                    )
                    
                    post_processed_image = post_process_image.remote(result["formats"]["original"])
                    
                    result["formats"]["svg"] = post_processed_image
                    print(f"✅ Converted to SVG")
                except Exception as e:
                    print(f"⚠️ Post-processing failed: {e}")
                    result["errors"] = result.get("errors", {})
                    result["errors"]["post_processing"] = "Failed to post-process icon"
                
                processed_images.append(result)
            
            elapsed = time.time() - start_time
            print(f"⚡⚡⚡ Complete pipeline executed in {elapsed:.2f} seconds")
            
            if result.get("errors"):
                self.update_job_status(
                    callback_url, 
                    JobStatus.FAILED, 
                    metadata,
                    "Job failed", 
                    error_details=result["errors"]
                )
            else:
                self.update_job_status(
                    callback_url, 
                    JobStatus.COMPLETED, 
                    metadata,
                    "Job completed successfully", 
                    processed_images
            )
            
            return {"images": processed_images}
        except Exception as e:
            import traceback
            import datetime
            
            error_traceback = traceback.format_exc()
            current_stage = "unknown"
            
            if "Loading model" in str(e):
                current_stage = JobStatus.INITIATED
            elif "inference" in str(e).lower():
                current_stage = JobStatus.INFERENCE
            elif "post-processing" in str(e).lower():
                current_stage = JobStatus.POST_PROCESSING
            
            error_details = {
                "message": str(e),
                "stage": current_stage,
                "traceback": error_traceback,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            print(f"❌ Error in job processing at stage '{current_stage}': {e}\n{error_traceback}")
            
            self.update_job_status(
                callback_url,
                JobStatus.FAILED, 
                metadata,
                f"Job failed during {current_stage}: {str(e)}",
                error_details=error_details
            )
            
            raise
         
    @method()
    def process_job_queue(self, prompt: str, config: dict, callback_url: str = None, metadata: dict = None):        
        inference_config = InferenceConfig(**config)
        
        if config.get("is_warmup", False):
            print("🔥 Processing warm-up job")
            self.ensure_models_loaded()
            return {"status": "warmed_up"}
        
        job_id = metadata.get("job_id")
        user_id = metadata.get("user_id")
        
        if not user_id:
            raise ValueError("User ID is required")
        
        if not job_id:
            raise ValueError("Job ID is required")
        
        if not callback_url:
            raise ValueError("Callback URL is required")
        
        return self.process_job(prompt, inference_config, job_id, callback_url, user_id)
    
