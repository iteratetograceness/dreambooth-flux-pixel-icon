from modal import App, Image as ModalImage, Volume, Secret

app = App("flux-pipeline-optimize")
huggingface_secret = Secret.from_name("huggingface-secret")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = ModalImage.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.10"
).entrypoint([])

diffusers_commit_sha = "33d10af28fcfb4d41ab7fb97d84c8ac2317576d5"

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
        "invisible_watermark",
        "transformers",
        "huggingface_hub[hf_transfer]",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "torch",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
        "torchao",
        "peft",
        "para-attn"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1", 
        "HF_HUB_CACHE": "/cache", 
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHAO_AUTOTUNER_ENABLE": "0",
        "TORCHAO_AUTOTUNER_DATA_PATH": "/quant/autotuner_data.pkl",
    })
)

inference_image = (
    ModalImage.from_registry(
        f"nvidia/cuda:{cuda_version}-runtime-{operating_sys}", 
        add_python="3.10"
    )
    .apt_install(
        "git",
    )
    .pip_install(
        "torch",
        "numpy<2",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "pypatch",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "para-attn",
        "torchao"
    )
    .add_local_file("./patches/flux_patch.patch", "/root/flux_patch.patch", copy=True)
    .run_commands(
        "pypatch apply /root/flux_patch.patch diffusers"
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHAO_AUTOTUNER_ENABLE": "0",
        # "TORCHAO_AUTOTUNER_DATA_PATH": "/quant/autotuner_data.pkl",
    })
)

volume = Volume.from_name("aot-compiled-flux-models", create_if_missing=True)
sample_image_volume = Volume.from_name("sample-image", create_if_missing=True)

@app.function(
    gpu="H100", 
    image=flux_image,
    volumes={
        "/aot_compiled_models": volume,
        "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": Volume.from_name(
            "triton-cache", create_if_missing=True
        ),
        "/root/.inductor-cache": Volume.from_name(
            "inductor-cache", create_if_missing=True
        ),
    },
    timeout=60 * 60 * 3,
    secrets=[huggingface_secret]
)
def optimize_transformer(
    model_path="graceyun/dotelier-color",
    batch_size=1,
    height=1024,
    width=1024,
    quant_type="int8wo",
):
    import gc
    import time
    import torch
    import os
    from diffusers import FluxTransformer2DModel, FluxPipeline, TorchAoConfig
    from torchao.utils import unwrap_tensor_subclass
    import torch.utils.benchmark as benchmark
    
    torch._inductor.config.mixed_mm_choice = "triton"
    output_dir = "/aot_compiled_models"
    
    def memory_cleanup():
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
            
    @torch.no_grad()
    def load_model():
        model = FluxTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        return model
    
    def time_fn(f, *args, **kwargs):
        begin = time.time()
        f(*args, **kwargs)
        end = time.time()
        print(f"Time: {end - begin:.2f}s")
        return end - begin
    
    def prepare_latents(batch_size, height, width, num_channels_latents=1):
        vae_scale_factor = 16
        height = 2 * (int(height) // vae_scale_factor)
        width = 2 * (int(width) // vae_scale_factor)
        shape = (batch_size, num_channels_latents, height, width)
        pre_hidden_states = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        hidden_states = FluxPipeline._pack_latents(
            pre_hidden_states, batch_size, num_channels_latents, height, width
        )
        return hidden_states
    def prepare_realistic_inputs():
        num_channels_latents = 64 // 4
        hidden_dim = 4096
        vae_scale_factor = 16
        h = 2 * (int(height) // vae_scale_factor)
        w = 2 * (int(width) // vae_scale_factor)
        shape = (batch_size, num_channels_latents, h, w)
        
        pre_hidden_states = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        hidden_states = FluxPipeline._pack_latents(
            pre_hidden_states, batch_size, num_channels_latents, h, w
        )
        
        num_img_sequences = hidden_states.shape[1]
        img_ids = torch.randn(num_img_sequences, 3, dtype=torch.bfloat16, device="cuda")
        txt_ids = torch.randn(512, 3, dtype=torch.bfloat16, device="cuda")
        pooled_projections = torch.randn(batch_size, 768, dtype=torch.bfloat16, device="cuda")
        

        timestep = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        guidance = torch.tensor([3.5], dtype=torch.float32, device="cuda")

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": torch.randn(batch_size, 512, hidden_dim, dtype=torch.bfloat16, device="cuda"),
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance,
            "joint_attention_kwargs": None,
            "return_dict": False
        }
        
        print("\nGenerated input tensors with shapes:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}, {value.dtype}")
                
        print(f"\nSaving inputs to {output_dir}...")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                save_path = f"{output_dir}/transformer_{key}.pt"
                torch.save(value, save_path)
                print(f"Saved {key} to {save_path}")
        
        memory_cleanup()
        return inputs
        
    def get_example_inputs(force_generate=False):
        if force_generate:
            return prepare_realistic_inputs()
            
        try:
            required_keys = [
                "hidden_states", "encoder_hidden_states", "pooled_projections", 
                "timestep", "img_ids", "txt_ids", "guidance"
            ]
            
            all_exist = all(
                os.path.exists(f"{output_dir}/transformer_{key}.pt") 
                for key in required_keys
            )
            
            if not all_exist:
                print("Some input files missing, generating new inputs")
                return prepare_realistic_inputs()
            
            example_inputs = {}
            for key in required_keys:
                tensor_path = f"{output_dir}/transformer_{key}.pt"
                example_inputs[key] = torch.load(tensor_path, map_location="cuda")
                print(f"Loaded {key}: {example_inputs[key].shape}, {example_inputs[key].dtype}")
            
            example_inputs["joint_attention_kwargs"] = None
            example_inputs["return_dict"] = False
            
            return example_inputs
            
        except Exception as e:
            print(f"Error loading example inputs: {e}")
            return prepare_realistic_inputs()
    
    def aot_compile(name, model, **sample_kwargs):
        config = torch._inductor.config
        config.disable_progress = False
        config.conv_1x1_as_mm = True
        config.coordinate_descent_tuning = True
        config.coordinate_descent_check_all_directions = True
        config.epilogue_fusion = False
    
        path = f"{output_dir}/{name}.pt2"
        options = {
            "max_autotune": True,
            "triton.cudagraphs": True,
        }
        print(f"Compiling {name} with {path}")
        print(f"Options: {options}")
        return torch._inductor.aoti_compile_and_package(
            torch.export.export(model, (), sample_kwargs),
            package_path=path,
            inductor_configs=options,
        )
    
    @torch.no_grad()
    def validate_model(model, inputs, name=""):
        with torch.inference_mode():
            outputs = model(**inputs)
            print(f"{name} output shape: {outputs[0].shape}")
            return outputs
        
    try:
        print("Loading model...")
        model = load_model()
        memory_cleanup()
        
        print("Preparing example inputs...")
        inputs = get_example_inputs()
        memory_cleanup()
        
        print("Unwrapping tensor subclasses...")
        unwrap_tensor_subclass(model)
        memory_cleanup()
        
        try:
            print("AOT compiling...")
            name = f"aot_transformer_0" # CHANGE NAME UPON EACH RUN
            path = aot_compile(name, model, **inputs)
            print(f"Compiled to: {path}")
        except Exception as e:
            print(f"Error during AOT compilation: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            memory_cleanup()
        
        print("Testing compiled model...")
        compiled_func = torch._inductor.aoti_load_package(path)
        compiled_outputs = validate_model(compiled_func, inputs, name="Compiled model")
        max_diff_compiled = (reference_outputs[0] - compiled_outputs[0]).abs().max().item()
        print(f"Max difference after compilation: {max_diff_compiled}")
        
        if max_diff_compiled > 0.1:
            print(f"WARNING: Large difference after compilation: {max_diff_compiled}")
            print("This may result in poor image quality")
        
        return {
            "compiled_path": path,
            "max_diff_after_fusion": float(max_diff_fused),
            "max_diff_after_compilation": float(max_diff_compiled),
            "success": True
        }
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        print(traceback.format_exc())
        return {"error": str(e), "success": False}
    finally:
        memory_cleanup()

@app.function(
    gpu="H100", # L40S, A100-40GB, A100-80GB
    image=inference_image,
    secrets=[huggingface_secret],
    volumes={
        "/aot_compiled_models": volume,
        "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": Volume.from_name("inductor-cache", create_if_missing=True),
        "/images": sample_image_volume,
        "/quant": Volume.from_name("quant-cache", create_if_missing=True),
    },
    timeout=1800, 
    allow_concurrent_inputs=5,
    concurrency_limit=5
)
def test_inference(
    prompt: str = 'a PXCON style icon of a cow, 16-bit',
    model_path: str = "graceyun/dotelier-color",
    compiled_path: str = '/aot_compiled_models/aot_transformer_0.pt2',
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    height: int = 1024,
    width: int = 1024,
    compile: bool = False,
    fbcache: bool = True,
    quant_transformer: bool = False,
    quant_vae: bool = True,
    
):
    import gc
    import time
    import torch
    import sys
    
    if compile:
        # Clear Python cache for the problematic module
        if 'diffusers.pipelines.flux.pipeline_flux' in sys.modules:
            del sys.modules['diffusers.pipelines.flux.pipeline_flux']
        
        # Remove *.pyc files to force recompilation
        import os
        import glob
        for pyc_file in glob.glob("/usr/local/lib/python3.10/site-packages/diffusers/pipelines/flux/__pycache__/pipeline_flux.cpython-*.pyc"):
            try:
                os.remove(pyc_file)
                print(f"Removed cache file: {pyc_file}")
            except Exception as e:
                print(f"Failed to remove cache file: {e}")
            
    from diffusers import FluxPipeline

    def memory_cleanup():
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
         
    try:        
        print("Loading pipeline...")
        
        print(f"Model path: {model_path}")
        print(f"Height: {height}, Width: {width}")

        pipeline = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        
        pipeline.enable_model_cpu_offload()
        pipeline.transformer.fuse_qkv_projections()
        
        if quant_transformer:
            import torchao
            import pickle
            from torchao.quantization.autoquant import AUTOQUANT_CACHE
            
            with open("/quant/transformer_autoquant_cache.pkl", "rb") as f:
                AUTOQUANT_CACHE.update(pickle.load(f))

            pipeline.transformer = torchao.autoquant(pipeline.transformer, error_on_unseen=False)

            AUTOQUANT_CACHE.clear()
        
        if quant_vae:
            import torchao
            import pickle
            from torchao.quantization.autoquant import AUTOQUANT_CACHE
            
            with open("/quant/vae_autoquant_cache.pkl", "rb") as f:
                AUTOQUANT_CACHE.update(pickle.load(f))
                
            pipeline.vae = torchao.autoquant(pipeline.vae, error_on_unseen=False)

        if compile:
            print("Loading compiled transformer...")
            compiled_transformer = torch._inductor.aoti_load_package(compiled_path)
            pipeline.transformer = compiled_transformer
            memory_cleanup()
            
        if fbcache:
            from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
            
            apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12)
                
        print(f"Running inference for prompt: {prompt}")
        _ = pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )
        memory_cleanup()
        
        # Timed run
        start_time = time.time()
        images = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000
        print(f"Inference time: {inference_time:.2f}ms")
        print(f"Generated image of size {height}x{width}")
            
        from datetime import datetime
        now = datetime.now()
        images[0].save(f"/images/test_qt{quant_transformer}_qv{quant_vae}_fb{fbcache}_{now.strftime('%Y%m%d_%H%M')}.png")
        
        memory_cleanup()
    except Exception as e:
        print(f"Error during inference: {e}")
        print(f"Error type: {type(e)}")

        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}
    finally:
        memory_cleanup()

@app.function(
    gpu="H100",
    image=inference_image,
    secrets=[huggingface_secret],
    volumes={
        "/aot_compiled_models": volume,
        "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": Volume.from_name("inductor-cache", create_if_missing=True),
        "/images": sample_image_volume,
    },
    timeout=1800,
)
def benchmark(
    prompt: str = 'a PXCON style icon of a cow, 16-bit',
    model_path: str = "graceyun/dotelier-color",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    height: int = 1024,
    width: int = 1024,
):
    import time
    import torch
    from diffusers import FluxPipeline
    
    base_pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    def time_fn(f, *args, **kwargs):
        begin = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"Time: {end - begin:.2f}s")
        return result
    
    print("Benchmarking original transformer...")
    def baseline():
        image = base_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images[0]
        return image
    time_fn(baseline)
    
    print("Benchmarking First Block Cache optimization")
    from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
    
    residual_diff_thresholds = [0.08, 0.10, 0.12]
    
    def first_block_cache(p):
            """
            ParaAttention directly uses the residual difference of the first transformer block output to approximate the difference among model outputs. When the difference is small enough, the residual difference of previous inference steps is reused. In other words, the denoising step is skipped.
            """
            image = p(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            ).images[0]
            return image
    
    for residual_diff_threshold in residual_diff_thresholds:
        print(f"First Block Cache: residual_diff_threshold={residual_diff_threshold}...")
        
        apply_cache_on_pipe(base_pipe, residual_diff_threshold=residual_diff_threshold)
        
        img = time_fn(first_block_cache, base_pipe)
        img.save(f"/images/fbcache_{residual_diff_threshold}.png")
 
@app.function(
    gpu="H100",
    image=flux_image,
    secrets=[huggingface_secret],
    volumes={
        "/cache": Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/images": sample_image_volume,
        "/quant": Volume.from_name("quant-cache", create_if_missing=True),
    },
    timeout=60 * 60 * 3,
)
def autoquant(
    prompt: str = 'a PXCON style icon of a cow, 16-bit',
    model_path: str = "graceyun/dotelier-color",
    guidance_scale: float = 5.0,
    height: int = 1024,
    width: int = 1024,
    type: str = "transformer",
):
    import torch
    import pickle
    from diffusers import FluxPipeline
    import torchao
    from torchao.quantization.autoquant import AUTOQUANT_CACHE

    # Enable SDPA optimization
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    print("Loading model...")
    pipeline = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    pipeline.transformer.fuse_qkv_projections()
    torch.cuda.empty_cache()
    AUTOQUANT_CACHE.clear()
    
    if type == "transformer":
        print("Applying autoquant to transformer...")
        pipeline.transformer = torchao.autoquant(pipeline.transformer, error_on_unseen=False)
        
        # Trigger benchmarking
        _ = pipeline(
            prompt, 
            guidance_scale=guidance_scale, 
            num_inference_steps=1,
            height=height,
            width=width
        )
        
        # Save transformer cache
        with open("/quant/transformer_autoquant_cache.pkl", "wb") as f:
            pickle.dump(AUTOQUANT_CACHE, f)
        
    if type == "vae":
        print("Applying autoquant to VAE...")
        pipeline.vae = torchao.autoquant(pipeline.vae, error_on_unseen=False)
        
        # Prepare VAE inputs
        vae_scale_factor = 16
        vae_h = height // vae_scale_factor  
        vae_w = width // vae_scale_factor
        latents = torch.randn(1, 16, vae_h, vae_w, device="cuda", dtype=torch.bfloat16)
        
        # Trigger benchmarking
        with torch.no_grad():
            _ = pipeline.vae.decode(latents)
        
        # Save VAE cache
        with open("/quant/vae_autoquant_cache.pkl", "wb") as f:
            pickle.dump(AUTOQUANT_CACHE, f)
    
@app.local_entrypoint()
def main():
    # autoquant.remote(type="transformer")
    test_inference.remote()
    