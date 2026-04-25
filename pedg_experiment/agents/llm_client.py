"""
LLM Client for PEDG - supports local vLLM + OpenAI-compatible API
"""
import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests
import openai

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    raw: Any
    model: str
    tokens_used: Optional[int] = None


class LLMWrapper:
    """Wrapper for LLM API calls with retry and caching"""
    
    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model_name: str = "Qwen3.5-9B",
        api_key: str = "EMPTY",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 60,
        cache_dir: str = "./results/cache",
    ):
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use OpenAI-compatible client
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=2,
        )
        
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _make_cache_key(self, messages: List[Dict], temperature: float) -> str:
        import hashlib
        content = json.dumps({"messages": messages, "temp": temperature}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _read_cache(self, cache_key: str) -> Optional[str]:
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    self._cache_hits += 1
                    return data["content"]
            except Exception:
                pass
        self._cache_misses += 1
        return None
    
    def _write_cache(self, cache_key: str, content: str):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, "w") as f:
                json.dump({"content": content}, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def chat(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        role_hint: str = "unknown",
    ) -> LLMResponse:
        """
        Make a chat completion call.
        messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
        """
        temp = temperature if temperature is not None else self.temperature
        maxt = max_tokens if max_tokens is not None else self.max_tokens
        
        # Check cache first
        cache_key = self._make_cache_key(messages, temp)
        cached = self._read_cache(cache_key)
        if cached:
            return LLMResponse(content=cached, raw=None, model=self.model_name)
        
        # Make API call with retries
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=maxt,
                    timeout=self.timeout,
                )
                
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens if hasattr(response, "usage") else None
                
                # Cache result
                self._write_cache(cache_key, content)
                
                return LLMResponse(
                    content=content,
                    raw=response,
                    model=self.model_name,
                    tokens_used=tokens,
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError(f"LLM call failed after {max_attempts} attempts: {last_error}")
    
    def parse_structured_response(self, text: str, fields: List[str]) -> Dict[str, str]:
        """Parse structured fields from LLM response text"""
        import re
        result = {}
        lines = text.split('\n')
        
        for field in fields:
            # Search for field name in the text
            field_lower = field.lower()
            found = False
            for line in lines:
                if line.lower().startswith(field_lower + ':'):
                    value_part = line.split(':', 1)[1].strip()
                    # Remove markdown formatting
                    value_part = re.sub(r'[\*_#]+', '', value_part).strip()
                    # For confidence, extract first integer
                    if field_lower == "confidence":
                        digits = re.search(r'\d+', value_part)
                        result[field_lower] = digits.group() if digits else "50"
                    else:
                        result[field_lower] = value_part
                    found = True
                    break
            
            if not found:
                result[field_lower] = ""
        
        return result
    
    def stats(self) -> Dict:
        return {"cache_hits": self._cache_hits, "cache_misses": self._cache_misses}
