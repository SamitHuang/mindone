from mindnlp.core import nn, ops
import re
import math
from mindnlp.transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
# from mindspore import mint


def build_vision_tower(vision_tower='internlm/internlm-xcomposer2d5-clip'):
    return CLIPVisionTower(vision_tower)


def build_vision_projector():
    projector_type = 'mlp2x_gelu'
    mm_hidden_size = 4096
    mid_hidden_size = 4096
    hidden_size = 4096

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, mid_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(mid_hidden_size, mid_hidden_size))

        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        #self.conv_dim = 8192
        #self.conv = nn.Conv2d(1024, self.conv_dim,3,2,1)
        self.select_layer = -1
        self.select_feature = 'patch'
        self.load_model()

    def load_model(self):
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def resize_pos(self):
        print ('Dummy Resized')

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, glb_GN, sub_GN):
        if not self.is_loaded:
            self.load_model()
        assert type(images) is list
        shapes = []
        input_imgs = []
        for img in images:
            _, C, H, W = img.shape
            shapes.append([H//560, W//560])
            sub_img = img.reshape(1,3,H//560,560,W//560,560).permute(0,2,4,1,3,5).reshape(-1,3,560,560).contiguous()
            glb_img = nn.functional.interpolate(img.float(), size=(560,560), mode='bicubic',).to(sub_img.dtype)
            input_imgs.append(glb_img)
            input_imgs.append(sub_img)
        input_imgs = ops.cat(input_imgs, dim=0)

        image_forward_outs = self.vision_tower(input_imgs.to(dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(input_imgs.dtype) ### B*?, N, C
        _, N, C = image_features.shape
        H = int(math.sqrt(N))
        assert N == 40 ** 2

        output_imgs = []
        output_len = []
        for [h, w] in shapes:
            B_ = h*w
            glb_img = image_features[:1] ### 1, N, C
            glb_img = glb_img.reshape(1,H,H,C).reshape(1,H//2,2,H//2,2,C).contiguous().permute(0,1,3,2,4,5).reshape(1,H//2,H//2,4*C).contiguous()
            # import pdb; pdb.set_trace()
            temp_glb_GN = ops.tile(sub_GN, (1, H//2, 1, 1))
            glb_img = ops.cat([glb_img, temp_glb_GN.to(glb_img.dtype)], dim=2).reshape(1,-1,4*C)
            
            sub_img = image_features[1:1+B_] ### ?, N, C
            sub_img = sub_img.reshape(B_,H,H,C).reshape(B_,H//2,2,H//2,2,C).contiguous().permute(0,1,3,2,4,5).reshape(B_,-1,4*C).contiguous()
            sub_img = sub_img.reshape(1, h, w, 20, 20, -1).permute(0,1,3,2,4,5).reshape(1,h*20,w*20,4*C)
            temp_sub_GN = ops.tile(sub_GN, (1, h*20, 1, 1))
            sub_img = ops.cat([sub_img, temp_sub_GN.to(sub_img.dtype)], dim=2).reshape(1,-1,4*C)

            output_imgs.append(ops.cat([glb_img, glb_GN.to(glb_img.dtype), sub_img], dim=1))
            temp_len = int((h*w+1)*400 + 1 + (h+1)*20)
            assert temp_len == output_imgs[-1].shape[1]
            output_len.append(temp_len)

            image_features = image_features[1+h*w:]

        output_imgs = ops.cat(output_imgs, dim=1)

        return output_imgs, output_len

    @property
    def dummy_feature(self):
        return ops.zeros(1, self.hidden_size, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
