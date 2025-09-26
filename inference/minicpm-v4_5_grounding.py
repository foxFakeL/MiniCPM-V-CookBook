import re
from PIL import Image, ImageDraw
import torch
from transformers import AutoModel, AutoTokenizer

def setup_model_and_tokenizer(model_path):
    dtype = torch.bfloat16
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model = model.to(dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def extract_bbox_from_response(response):
    match = re.search(r"<box>([\d\s]+)</box>", response)
    if match:
        bbox_str = match.group(1)
        bbox = list(map(int, bbox_str.strip().split()))
        return bbox
    else:
        raise ValueError("Can't find bbox in response")

def draw_bbox_on_image(image, bbox):
    w, h = image.size
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    return image

def model_infer_and_draw(img_path, question, model, tokenizer):
    image = Image.open(img_path)
    msgs = [
        {'role': 'user', 'content': [question, image]},
    ]
    with torch.inference_mode():
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=1024,
            max_inp_length=8192,
            use_image_id=True,
        )
    print("Model output:", res)
    bbox = extract_bbox_from_response(res)
    image_with_box = draw_bbox_on_image(image, bbox)
    return res, bbox, image_with_box

model_path = 'openbmb/MiniCPM-V-4_5'
img_path = './assets/airplane.jpeg'
question = 'Please provide the bounding box coordinate of the region this sentence describes: <ref>airplane</ref>'

model, tokenizer = setup_model_and_tokenizer(model_path)
res, bbox, image_with_box = model_infer_and_draw(img_path, question, model, tokenizer)

out_path = './assets/airplane_grounding.jpeg'
image_with_box.save(out_path)