# clip_helpers.py
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = "ViT-B/32"
model, preprocess = clip.load(clip_model, device=device)
clip_dim = 512  # Default dimension for ViT-B/32

def encode_query_text_with_clip(query_text):
    text = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        query_vector_image = model.encode_text(text).cpu().numpy()
    return query_vector_image
