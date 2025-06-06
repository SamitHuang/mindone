from mindone.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import mindspore as ms
from mindspore import tensor

ms.set_context(mode=1)

model_name = "/home/hyx/models/Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
# model = ms.jit(model.construct, dynamic=1)
# compiled_model = torch.compile(model, dynamic=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="np")


generated_ids = model.generate(
    tensor(model_inputs.input_ids),
    # **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

