"""Test: Qwen2.5-VL via vLLM .chat() API."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from vllm import LLM, SamplingParams

print("Loading Qwen2.5-VL...")
llm = LLM(
    model='/home/jinxu/models/Qwen/Qwen2___5-VL-3B-Instruct',
    trust_remote_code=True,
    gpu_memory_utilization=0.80,
    max_model_len=8192,
    limit_mm_per_prompt={'image': 1},
)
print("Loaded!")

# Try the .chat() API which has built-in chat template handling
sampling = SamplingParams(max_tokens=50, temperature=0.0)

# Text-only
print("\n=== Test 1: text-only ===")
out = llm.chat([{"role": "user", "content": "What is 2+2?"}], sampling_params=sampling)
print("Result:", out[0].outputs[0].text)

# Vision test with base64 image
import base64
png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
image_url = f"data:image/png;base64,{png_b64}"

print("\n=== Test 2: vision (base64 red pixel) ===")
out = llm.chat([{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "What color is this image?"},
    ]
}], sampling_params=sampling)
print("Result:", out[0].outputs[0].text)
print("SUCCESS!")
