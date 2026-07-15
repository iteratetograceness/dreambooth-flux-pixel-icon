from modal import App, Volume
import os

app = App("dotelier-config-setup")
sample_images = Volume.from_name("dotelier-sample-images", create_if_missing=True)

# Static configuration constants
DEFAULT_STYLE = "color"
NUM_INFERENCE_STEPS = 60
GUIDANCE_SCALE = 5
NUM_OUTPUTS = 1

STYLE_CONFIGS = {
    "color": {
        "token": "PXCON",
        "suffix": ", pixel art, 16-bit style, clean minimal design, white background, crisp black outline",
        "negative_prompt": "ugly, blurry, noisy, messy, dirty, complex, detailed, photorealistic, 3d, gradient, shading, multiple outlines, double outline, text that says pxcon, text that says PXCON, written text, signature, watermark, low quality, distorted, deformed",
    }
}

ALLOWED_ORIGINS = [
    "https://dotelier.studio",
    "https://www.dotelier.studio",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Secret comes from the environment (Modal Secret "vercel-automation-bypass").
# Never hardcode a fallback here — the previous default is in git history and
# must be rotated in Vercel.
VERCEL_AUTOMATION_BYPASS_SECRET = os.getenv("VERCEL_AUTOMATION_BYPASS_SECRET")

# Configuration class for easy access
class Config:
    DEFAULT_STYLE = DEFAULT_STYLE
    NUM_INFERENCE_STEPS = NUM_INFERENCE_STEPS
    GUIDANCE_SCALE = GUIDANCE_SCALE
    NUM_OUTPUTS = NUM_OUTPUTS
    STYLE_CONFIGS = STYLE_CONFIGS
    ALLOWED_ORIGINS = ALLOWED_ORIGINS
    VERCEL_AUTOMATION_BYPASS_SECRET = VERCEL_AUTOMATION_BYPASS_SECRET

@app.function()
def get_config():
    """Return the current configuration"""
    return {
        "DEFAULT_STYLE": DEFAULT_STYLE,
        "NUM_INFERENCE_STEPS": NUM_INFERENCE_STEPS,
        "GUIDANCE_SCALE": GUIDANCE_SCALE,
        "NUM_OUTPUTS": NUM_OUTPUTS,
        "STYLE_CONFIGS": STYLE_CONFIGS,
        "ALLOWED_ORIGINS": ALLOWED_ORIGINS,
        # secret deliberately not returned here
    }

@app.local_entrypoint()
def main():
    # Just print the config to verify it's working
    config = get_config.remote()
    print("Configuration loaded:")
    print(f"Default style: {config['DEFAULT_STYLE']}")
    print(f"Inference steps: {config['NUM_INFERENCE_STEPS']}")
    print(f"Guidance scale: {config['GUIDANCE_SCALE']}")
    print(f"Style configs: {list(config['STYLE_CONFIGS'].keys())}")
    print(f"Secret loaded from env: {'VERCEL_AUTOMATION_BYPASS_SECRET' in os.environ}")
