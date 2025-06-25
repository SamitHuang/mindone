# models
#### Here we provide common AIGC models. Stay tuned!


#### SigLIP Usage

Install mindone:

```
git clone -b fix_js --single-branch https://github.com/SamitHuang/mindone.git
cd mindone
pip install -e .

```

Simple usage in code:

```python
from mindone.models.siglip_vit import create_model

model = create_model(
    model_name,
    param_dtype=ms.bfloat16,
)

```
