import os
from flask import Flask, render_template, request
import torch
import random
import numpy as np
from utils import header, add_content, box
from wtforms import Form, TextField, IntegerField, DecimalField, BooleanField, validators, SubmitField, SelectField
from wtforms.fields.html5 import IntegerRangeField
import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel
import generate

# copied from generate.py
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                    help='模型参数')
# moved to model directory
# parser.add_argument('--tokenizer_path', default='cache/vocab_processed.txt', type=str, required=False, help='词表路径')
# parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
parser.add_argument('--prefix', default='豬肉', type=str, required=False, help='生成文章的开头')
parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
parser.add_argument('--segment', action='store_true', help='中文以词为单位')
parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

args = parser.parse_args()
print('args:\n' + args.__repr__())

if args.segment:
    from tokenizations import tokenization_bert_word_level as tokenization_bert
else:
    from tokenizations import tokenization_bert

os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
batch_size = args.batch_size
repetition_penalty = args.repetition_penalty

device = "cuda" if torch.cuda.is_available() else "cpu"

current_modelname = None
current_model = None
current_tokenizer = None
def getModel(modelname):
    global current_modelname, current_model, current_tokenizer
    if current_modelname != modelname:
        print(f"loading {modelname}")
        current_modelname = modelname

        # Code used before adding a models directory to support multiple models (which may have different vocab)
        # parser.add_argument('--tokenizer_path', default='cache/vocab_processed.txt', type=str, required=False, help='词表路径')
        # tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
        tokenizer_path = f'models/{modelname}/cache/vocab_processed.txt'
        current_tokenizer = tokenization_bert.BertTokenizer(vocab_file=tokenizer_path)
        print(f"tokenizer loaded from {tokenizer_path}")

        # Code used before adding a models directory to support multiple models
        # parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
        # model = GPT2LMHeadModel.from_pretrained(args.model_path)
        model_path = f'models/{modelname}/model'
        current_model = GPT2LMHeadModel.from_pretrained(model_path)
        current_model.to(device)
        current_model.eval()
        print(f"model loaded from {model_path}")

    return current_model, current_tokenizer

def text_generator(seed, length, temperature, topk, topp, fast_pattern, nsamples, modelname):
    print(f"""
seed: {seed}
length: {length}
temperature: {temperature}
topk: {topk}
topp: {topp}
fast_pattern: {fast_pattern}
modelname: {modelname}""")

    model, tokenizer = getModel(modelname)

    n_ctx = model.config.n_ctx

    samples_combined = ''
    samples_list = []

    if length == -1:
        length = model.config.n_ctx

    # generate text
    raw_text = seed
    context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
    generated = 0
    for _ in range(nsamples // batch_size):
        out = generate.generate(
            n_ctx=n_ctx,
            model=model,
            context=context_tokens,
            length=length,
            is_fast_pattern=fast_pattern, tokenizer=tokenizer,
            temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty, device=device
        )
        for i in range(batch_size):
            generated += 1
            text = tokenizer.convert_ids_to_tokens(out)
            for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                if generate.is_word(item) and generate.is_word(text[i + 1]):
                    text[i] = item + ' '
            for i, item in enumerate(text):
                if item == '[MASK]':
                    text[i] = ''
                elif item == '[CLS]':
                    text[i] = '\n\n'
                elif item == '[SEP]':
                    text[i] = '\n'
            info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
            print(info)
            text = ''.join(text).replace('##', '').strip()
            print(text)

            samples_combined = samples_combined + info + text + "\n"
            samples_list.append(text)
    print("=" * 80)
    samples_combined = samples_combined + "=" * 80 + "\n"
    return samples_combined, samples_list


# Create app
app = Flask(__name__)
app.config['DEBUG'] = True

def checkModelDirectory(modelname):
    if not os.path.exists(f'models/{modelname}/cache/vocab_processed.txt'):
        print(f"!!!!!! WARNING: skipping model {modelname}: models/{modelname}/cache/vocab_processed.txt not found")
        return False
    if not os.path.exists(f'models/{modelname}/model/config.json'):
        print(f"!!!!!! WARNING: skipping model {modelname}: models/{modelname}/model/config.json not found")
        return False
    if not os.path.exists(f'models/{modelname}/model/pytorch_model.bin'):
        print(f"!!!!!! WARNING: skipping model {modelname}: models/{modelname}/model/pytorch_model.bin not found")
        return False
    return True

def findModels():
    return [f.name for f in os.scandir('models') if f.is_dir() and checkModelDirectory(f.name)]

class ReusableForm(Form):
    """User entry form for entering specifics for generation"""
    # Starting seed
    seed = TextField("請輸入一個起始句子:", default="豬肉", validators=[validators.InputRequired()])
  
    # Configure GPT2
    length = IntegerField("生成長度 (<=1024):", default=50, validators=[validators.InputRequired(), validators.NumberRange(-1, 1024)])
    temperature = DecimalField("文章生成的隨機度 (0.1-3):", default=2, places=1, validators=[validators.InputRequired(), validators.NumberRange(0.1, 3)])
    topk = IntegerField("前k大的機率頻繁詞抽樣:", default=12, validators=[validators.InputRequired()])
    topp = DecimalField("前k大的累積機率頻繁詞抽樣:", default=3, validators=[validators.InputRequired()])
    temperature = DecimalField("文章生成的隨機度 (0.1-3):", default=2, places=1, validators=[validators.InputRequired(), validators.NumberRange(0.1, 3)])
    fast_pattern = BooleanField("采用更加快的方式生成文本:", default=False)
    nsamples = IntegerRangeField('生成幾個樣本:', default=1)
    modelname = SelectField('模型')

    # Submit button
    submit = SubmitField("開始產生文章")



# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)

    # Determine model choices on every request so we don't need to
    # restart the server when adding or removing models
    form.modelname.choices = findModels()
    form.modelname.default = form.modelname.choices[0]

    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        # Extract information
        seed = request.form['seed']
        length = int(request.form['length'])
        temperature = float(request.form['temperature'])
        topk = int(request.form['topk'])
        topp = float(request.form['topp'])
        fast_pattern = 'fast_pattern' in request.form.keys()
        nsamples = int(request.form['nsamples'])
        modelname = request.form['modelname']

        # Generate a random sequence
        samples_combined, samples_list = text_generator(seed=seed,
                                                        length=length,
                                                        temperature=temperature,
                                                        topk=topk,
                                                        topp=topp,
                                                        fast_pattern=fast_pattern,
                                                        nsamples=nsamples,
                                                        modelname=modelname)
        return render_template('seeded.html',
                                seed=seed,
                                samples_combined=samples_combined,
                                samples_list=samples_list)
    else:
        # Send template information to index.html
        return render_template('index.html', form=form)


if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
           "please wait until server has fully started"))
 
    # Run app
    app.run(host="0.0.0.0", port=8000)

    # print(text_generator('豬肉')[0])
