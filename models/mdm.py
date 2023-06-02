import clip
import numpy as np
import torch as th
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class TextEmbedder():
    def __init__(self, device="cpu", clip_model="ViT-B/32"):
        self.device = device
        self.text_encoder_model = self._load_clip(clip_model, device)

    def _load_clip(self, clip_model, device):
        model, preprocess = clip.load(clip_model, device)
        model.eval()
        model.requires_grad_(False)
        return model
        
    def _encode_texts(self, texts):
        tokens = th.cat([clip.tokenize(text) for text in texts]).to(self.device)
        print(tokens)
        text_features = self.text_encoder_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features 

class MDM(nn.Module):
    def __init__(self, 
                 # using cuda
                 device="cpu",
                 # clip model
                 clip_model="ViT-B/32", 
                 ):
        super().__init__()
        

if __name__ == "__main__":
    device = "cuda" if th.has_cuda else "cpu"
    mdm = TextEmbedder()
    texts = ["running when raising hand"]
    print(mdm._encode_texts(texts)[0])
