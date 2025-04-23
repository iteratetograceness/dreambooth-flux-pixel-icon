from modal import Dict, App, Volume, Image

app = App("dotelier-config-setup")
inference_config = Dict.from_name("dotelier-inference-config", create_if_missing=True)
styles = Dict.from_name("dotelier-styles", create_if_missing=True)
http_config = Dict.from_name("dotelier-http-config", create_if_missing=True)
http_config_dev = Dict.from_name("dotelier-http-config-dev", create_if_missing=True)
sample_images = Volume.from_name("dotelier-sample-images", create_if_missing=True)

DEFAULT_STYLE = "color"
NUM_INFERENCE_STEPS = 60
GUIDANCE_SCALE = 5
NUM_OUTPUTS = 1
STYLE_CONFIGS = {}
ALLOWED_ORIGINS = [
    "https://dotelier.studio",
    "https://www.dotelier.studio",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

@app.function()
def setup_config():
    # import os
    inference_config["DEFAULT_STYLE"] = DEFAULT_STYLE
    # inference_config["NUM_INFERENCE_STEPS"] = NUM_INFERENCE_STEPS
    # inference_config["GUIDANCE_SCALE"] = GUIDANCE_SCALE
    # inference_config["NUM_OUTPUTS"] = NUM_OUTPUTS
    # styles["color"] = STYLE_CONFIGS["color"]
    # http_config["ALLOWED_ORIGINS"] = ALLOWED_ORIGINS
    # http_config["VERCEL_AUTOMATION_BYPASS_SECRET"] = ""
    print("config setup complete")

@app.local_entrypoint()
def main():
    setup_config.remote()
