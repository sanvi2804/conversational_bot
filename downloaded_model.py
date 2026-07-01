from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="./models/all-MiniLM-L6-v2"
)

from transformers import CLIPModel, CLIPProcessor


# Download CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Save CLIP locally
clip_model.save_pretrained("./models/clip")
clip_processor.save_pretrained("./models/clip")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



from transformers import CLIPModel, CLIPProcessor

CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import whisper
whisper.load_model("base")

from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
model.save("./models/reranker")

print("✅ Model loaded successfully!")