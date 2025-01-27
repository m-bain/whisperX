from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')

# audio files are decoded on the fly
inputs = feature_extractor(torch.randn((16000)), return_tensors="pt")
embeddings = model(**inputs).embeddings
embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

print(embeddings.shape)  # torch.Size([1, 768])