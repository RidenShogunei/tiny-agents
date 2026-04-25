"""Quick test: can vLLM load Qwen2.5-VL and generate text?"""
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

# Text-only test
messages = [{"role": "user", "content": "What is 2+2?"}]
sampling = SamplingParams(max_tokens=50, temperature=0.0)
outputs = llm.generate(messages, sampling_params=sampling)
print("Text-only result:", outputs[0].outputs[0].text)

# Image URL test (base64 PNG)
import base64
# 1x1 red PNG
png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
image_url = f"data:image/png;base64,{png_b64}"

messages = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "What color is this image?"},
    ]}
]
sampling = SamplingParams(max_tokens=30, temperature=0.0)
outputs = llm.generate(messages, sampling_params=sampling)
print("Vision result:", outputs[0].outputs[0].text)
print("SUCCESS: Qwen2.5-VL is working!")
