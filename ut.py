import math
import base64
import hashlib
import time
import hmac
from urllib.parse import urlencode
from typing import Callable, Dict
from sqids import Sqids
from sqids.constants import DEFAULT_ALPHABET

def djb2(s: str) -> int:
  h = 5381
  for char in reversed(s):
      h = (h * 33) ^ ord(char)
      # 32-bit integer overflow
      h &= 0xFFFFFFFF
  h = (h & 0xBFFFFFFF) | ((h >> 1) & 0x40000000)
  # Convert to signed 32-bit integer
  if h >= 0x80000000:
      h -= 0x100000000
  return h

def shuffle(string: str, seed: str) -> str:
  chars = list(string)
  seed_num = djb2(seed)
  for i in range(len(chars)):
      j = int(math.fmod(math.fmod(seed_num, i + 1) + i, len(chars)))
      chars[i], chars[j] = chars[j], chars[i]
  return "".join(chars)

def generate_key(file_seed: str, app_id: str) -> str:
  alphabet = shuffle(DEFAULT_ALPHABET, app_id)
  encoded_app_id = Sqids(alphabet, min_length=12).encode(
    [abs(djb2(app_id))]
  )
  return encoded_app_id + file_seed

def hmac_sha256(url: str, api_key: str) -> str:
    message = url.encode('utf-8')
    signature = hmac.new(api_key.encode('utf-8'), message, hashlib.sha256).hexdigest()
    return signature

def generate_presigned_url(
    app_id: str,
    api_key: str,
    file_name: str, 
    file_size: int,
    file_type: str = "image/png",
    ) -> Dict[str, str]:
    # Generate a unique seed for this file
    file_seed = base64.urlsafe_b64encode(
        hashlib.md5(file_name.encode() + str(time.time()).encode()).digest()
    ).decode()
        
    file_key = generate_key(file_seed, app_id)
    
    params = {
        "expires": str(int(time.time() * 1000) + 3600 * 1000),
        "x-ut-identifier": app_id,
        "x-ut-file-name": file_name,
        "x-ut-file-size": str(file_size),
        "x-ut-file-type": file_type,
    }
    
    base_url = f"https://sea1.ingest.uploadthing.com/{file_key}"
    
    url_with_params = f"{base_url}?{urlencode(params)}"
    
    signature = hmac_sha256(url_with_params, api_key)
    
    final_url = f"{url_with_params}&signature=hmac-sha256={signature}"
    
    return {
        "url": final_url,
        "file_key": file_key,
    }

def with_retry(
    func: Callable, 
    retries: int = 3, 
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,)
):
    import random
    for attempt in range(retries):
        try:
            return func()
        except exceptions as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} retries. Final error: {str(e)}")
                raise e
                
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.1), max_delay)
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            
    raise Exception(f"Failed after {retries} retries") # Shouldn't reach here
