"""
@file   clip_loss.py
@author Qiusheng Huang, Shanghai AI Lab
@brief  CLIP guidance loss
"""

import torch
import torch.nn as nn

from nr3d_lib.fmt import log

try:
    import clip
except ImportError:
    log.info("clip is not installed. CLIP related model & losses are disabled.")

class CLIPLoss(nn.Module):

    def __init__(self, img_len, text, w, path="/mnt/petrelfs/huangqiusheng/pretrained_models/ViT-B-32.pt"):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load(path, device="cuda")
        self.upsample = nn.Upsample(scale_factor=7)
        self.avg_pool = nn.AvgPool2d(kernel_size=128 // (32//(img_len//128))) # resize img to 224
        self.text = clip.tokenize(text).cuda()
        self.img_len = img_len
        self.w = w

    def forward(self, image, text=None):
        if len(image.shape)==3:
            image = torch.unsqueeze(image, 0)
        image = torch.reshape(image, [1,3,self.img_len,self.img_len])
        if text is None:
            text = self.text
        
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity*self.w
    
    def dis_imgs(self, image_s, image_t):
        # used by semantic infos guidance for novel views
        
        image_s = self.avg_pool(self.upsample(image_s))
        image_t = self.avg_pool(self.upsample(image_t))
        
        img_s_features = self.img_enc(image_s)
        img_t_features = self.img_enc(image_t)

        # normalized features
        img_s_features = img_s_features / img_s_features.norm(dim=1, keepdim=True)
        img_t_features = img_t_features / img_t_features.norm(dim=1, keepdim=True)

        # accumulate the features 
        if self.direction_mean is None:
            self.direction_mean = img_t_features
        else:
            self.direction_mean += img_t_features
        
        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * img_s_features @ self.direction_mean.t()

        similarity = 1.0 - logits_per_image / 100.0

        return similarity

    def img_enc(self, image):
        return self.model.encode_image(image)

    def text_enc(self, text):
        return self.model.encode_image(text)