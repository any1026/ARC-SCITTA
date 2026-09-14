import torch.nn.functional as F
import torch
import torch.nn as nn
import json
import math
import numpy as np
from scipy import ndimage

class TTA(nn.Module):
    """TTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, model_anchor, shape_stats_path):
        super().__init__()
        self.model = model
        self.model_anchor = model_anchor.eval()

        # Fundus-doFE uses background, optic disc rim, and optic cup.
        self.num_classes = 3
        self.max_lens = 40
        self.topk = 5
        self.threshold = 0.9
        # Fixed-length CCD history, matching the paper's SFT score queue S.
        self.entropy_list = []
        self.pool = Prototype_Pool(0.1,class_num=self.num_classes,max = self.max_lens).cuda()
        with open(shape_stats_path, 'r', encoding='utf-8') as f:
            self.shape_stats = json.load(f)
        self.ccd_pass = 0
        self.anatomy_pass = 0
        self.pool_accept = 0
    def forward(self, x, names):
        for _ in range(1):
            outputs = self.forward_and_adapt(x, self.model, names)
        return outputs

    torch.autograd.set_detect_anomaly(True)
    @torch.no_grad()
    def forward_and_adapt(self, x, model, names):
        layer_fea = 'med'
        self.ww = x.shape[-1]
        bad_num = x.shape[0]
        topk = self.topk
        latent_model = model.get_feature(x, loc = layer_fea)
        b,c,w,h = latent_model.shape
        sup_pixel = w
        latent_model = latent_model.reshape(b,c,int(w/sup_pixel),sup_pixel,int(h/sup_pixel),sup_pixel)
        latent_model = latent_model.permute(0,2,4,1,3,5)
        latent_model = latent_model[0].reshape(1*int(w/sup_pixel)*int(h/sup_pixel),c*sup_pixel*sup_pixel)
        latent_model_,fff, out_image, out_mask, len_pool = self.pool.get_pool_feature(latent_model,None,top_k = topk)

        with torch.no_grad():
            if len_pool < topk:
                threshold = len_pool / topk
            else:
                threshold = self.threshold
            # Option 1: Use the adapted model for CCD (works similarly in practice, since CCD mainly selects reliable samples)
            # fine = self.get_fine_ccd(x, model.eval(), self.entropy_list, threshold=threshold)

            # Option 2 (default, consistent with paper): Use the source model (model_anchor) for CCD, which retains source BN statistics
            fine = self.get_fine_ccd(x, self.model_anchor.eval(), self.entropy_list, threshold=threshold)

        # Anatomy-aware admission is an additional safety gate only. The
        # official CCD, SFT pool, SABE, and SFF computations above and below
        # are kept unchanged.
        if fine:
            self.ccd_pass += 1
            with torch.no_grad():
                anchor_prediction = self.model_anchor.eval()(x).argmax(1)
            anatomy_fine = self.get_fine_anatomy(anchor_prediction)
            if anatomy_fine:
                self.anatomy_pass += 1
            fine = fine and anatomy_fine

        if fine:
            self.pool.update_feature_pool(latent_model)
            self.pool.update_image_pool(x)
            self.pool.update_mask_pool(model(x).softmax(1))
            self.pool.update_name_pool(names[0])
            self.pool_accept += 1
        if out_image is not None:
            out_image = out_image[0]
            x_hised = torch.cat((x, out_image), dim=0)
            latent_model = model.get_feature(x_hised, loc = layer_fea)
            latent_model_ = latent_model_.view(bad_num,int(w/sup_pixel),int(h/sup_pixel),c,sup_pixel,sup_pixel)
            latent_model_ = latent_model_.permute(0,3,1,4,2,5)
            latent_model_ = latent_model_.reshape(bad_num,c,w,h)
            latent_model[0:1] = latent_model_
            outputs2 = model.get_output(latent_model,loc = layer_fea)[0:1].softmax(1)
            return outputs2
        else:
            return self.model_anchor(x)

    def diagnostics(self):
        return {
            'ccd_pass': int(self.ccd_pass),
            'anatomy_pass': int(self.anatomy_pass),
            'pool_accept': int(self.pool_accept),
            'pool_size': int(self.pool.feature_bank.shape[0]),
        }

    def get_fine_anatomy(self, prediction):
        """Reject anatomically implausible SFT candidates without target labels.

        Class 1 is the optic-disc rim and class 2 is the optic cup in the
        public Fundus encoding. The complete disc is therefore class 1 union
        class 2. Bounds are estimated only from source-domain masks.
        """
        pred = prediction[0].detach().cpu().numpy()
        disc = pred > 0
        cup = pred == 2
        if not disc.any() or not cup.any():
            return False

        def largest_fraction(mask):
            labels, count = ndimage.label(mask)
            if count == 0:
                return 0.0
            sizes = np.bincount(labels.ravel())[1:]
            return float(sizes.max() / max(1, mask.sum()))

        disc_area = float(disc.sum())
        cup_area = float(cup.sum())
        disc_center = np.asarray(ndimage.center_of_mass(disc), dtype=np.float64)
        cup_center = np.asarray(ndimage.center_of_mass(cup), dtype=np.float64)
        disc_radius = math.sqrt(disc_area / math.pi)
        values = {
            'cup_disc_ratio': cup_area / disc_area,
            'center_distance_norm': float(np.linalg.norm(cup_center - disc_center) / max(disc_radius, 1e-6)),
            'disc_lcc_fraction': largest_fraction(disc),
            'cup_lcc_fraction': largest_fraction(cup),
        }
        for name, value in values.items():
            bounds = self.shape_stats['admission_bounds'][name]
            if not (bounds['low'] <= value <= bounds['high']):
                return False
        return True

    def entropy(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p, dim=1)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en

    def get_fine_ccd(self, x, model_anchor, entropy_list, threshold=0.9):
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    pred1 = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = pred1.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = torch.randperm(pred1.size(0))
                select_point = 200
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                pred1_en = self.entropy(torch.matmul(pred1.t(), pred1))
                # The previous v1 reassigned a local slice, so the caller's
                # history grew to the whole stream instead of remaining L=40.
                # Store scalars and mutate the shared list in place.
                entropy_list.append(float(pred1_en.item()))
                if len(entropy_list) > self.max_lens:
                    del entropy_list[:-self.max_lens]
        sorted_list = sorted(entropy_list)
        ten_percent_index = int(len(sorted_list) * (1 - threshold))
        if ten_percent_index>0:
            ten_percent_min_value = sorted_list[:ten_percent_index][-1]
            return float(pred1_en.item()) <= ten_percent_min_value
        else:
            return False

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model



class Prototype_Pool(nn.Module):
    def __init__(self, delta=0.1, class_num=10, max=50):
        super(Prototype_Pool, self).__init__()
        self.class_num=class_num
        self.max_length = max
        self.feature_bank = torch.tensor([]).cuda()
        self.image_bank = torch.tensor([]).cuda()
        self.mask_bank = torch.tensor([]).cuda()
        self.name_list = []
    def get_pool_feature(self, x, mask, top_k = 5):
        if len(self.feature_bank)>0:
            cosine_similarities = torch.nn.functional.cosine_similarity(x.unsqueeze(1), self.feature_bank.unsqueeze(0), dim=2)
            if self.feature_bank.shape[0] >= top_k:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :top_k]
            else:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :self.feature_bank.shape[0]]
            rates = cosine_similarities[0][outall[0]].mean(0)
            weight = rates * torch.exp(cosine_similarities[0][outall[0]]) / torch.sum(torch.exp(cosine_similarities[0][outall[0]]))
            x = x * (1-rates)
            for i in range(min(top_k,self.feature_bank.shape[0])):
                x += self.feature_bank[outall[:,i]]*weight[i]
            return x,self.feature_bank[outall[:,]],self.image_bank[outall[:,]],self.mask_bank[outall[:,]], len(self.feature_bank)
        else:
            return x,x,None,None, len(self.feature_bank)

    def update_feature_pool(self, feature):
        feature = feature.detach()
        if self.feature_bank.numel() == 0:
            self.feature_bank = feature.clone()
        else:
            self.feature_bank = torch.cat([self.feature_bank, feature], dim=0)[-self.max_length:]
    def update_image_pool(self, image):
        image = image.detach()
        if self.image_bank.numel() == 0:
            self.image_bank = image.clone()
        else:
            self.image_bank = torch.cat([self.image_bank, image], dim=0)[-self.max_length:]
    def update_mask_pool(self, image):
        image = image.detach()
        if self.mask_bank.numel() == 0:
            self.mask_bank = image.clone()
        else:
            self.mask_bank = torch.cat([self.mask_bank, image], dim=0)[-self.max_length:]
    def update_name_pool(self, image):
        self.name_list.append(image)
        if len(self.name_list) > self.max_length:
            del self.name_list[:-self.max_length]
