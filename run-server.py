import os
from flask import Flask, render_template, request, Response
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
import urllib

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

def findModels():
    models = [f.name for f in os.scandir('models') if f.is_dir() and checkModelDirectory(f.name)]
    models.sort()
    return models

def checkModelDirectory(modelname):
    if not os.path.exists(f'models/{modelname}/cache/vocab_processed.txt'):
        app.logger.warning(f"!!!!!! WARNING: skipping model {modelname}: models/{modelname}/cache/vocab_processed.txt not found")
        return False
    if not os.path.exists(f'models/{modelname}/final_model/config.json'):
        app.logger.warning(f"!!!!!! WARNING: skipping model {modelname}: models/{modelname}/final_model/config.json not found")
        return False
    if not os.path.exists(f'models/{modelname}/final_model/pytorch_model.bin'):
        app.logger.warning(f"!!!!!! WARNING: skipping model {modelname}: models/{modelname}/final_model/pytorch_model.bin not found")
        return False
    return True

def getModelModificationTime(modelname):
    return [
        os.path.getmtime(f'models/{modelname}/cache/vocab_processed.txt'),
        os.path.getmtime(f'models/{modelname}/final_model/config.json'),
        os.path.getmtime(f'models/{modelname}/final_model/pytorch_model.bin')
    ]

current_modelname = None
current_modeltime = None
current_model = None
current_tokenizer = None
def getModel(modelname):
    global current_modelname, current_modeltime, current_model, current_tokenizer
    if current_modelname != modelname or current_modeltime != getModelModificationTime(modelname):
        app.logger.info(f"loading {modelname}")
        current_modelname = modelname
        current_modeltime = getModelModificationTime(modelname)

        # Code used before adding a models directory to support multiple models (which may have different vocab)
        # parser.add_argument('--tokenizer_path', default='cache/vocab_processed.txt', type=str, required=False, help='词表路径')
        # tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
        tokenizer_path = f'models/{modelname}/cache/vocab_processed.txt'
        current_tokenizer = tokenization_bert.BertTokenizer(vocab_file=tokenizer_path)
        app.logger.info(f"tokenizer loaded from {tokenizer_path}")

        # Code used before adding a models directory to support multiple models
        # parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
        # model = GPT2LMHeadModel.from_pretrained(args.model_path)
        model_path = f'models/{modelname}/final_model'
        current_model = GPT2LMHeadModel.from_pretrained(model_path)
        current_model.to(device)
        current_model.eval()
        app.logger.info(f"model loaded from {model_path}")

    return current_model, current_tokenizer

def text_generator(seed, length, temperature, topk, topp, fast_pattern, nsamples, modelname):
    app.logger.info(f"""
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
    prev_is_word = False
    for _ in range(nsamples):
        text = '' # This is only here so we can show the generated text in the log, but not used for the website.
        yield "[SAMPLE]"
        for token_id in generate.generate_stream(
                n_ctx=n_ctx,
                model=model,
                context=context_tokens,
                length=length,
                is_fast_pattern=fast_pattern, tokenizer=tokenizer,
                temperature=temperature, top_k=topk, top_p=topp, repetition_penalty=repetition_penalty, device=device):
            token = tokenizer.convert_ids_to_tokens(token_id)
            if prev_is_word and generate.is_word(token): # 确保英文前后有空格
                token = ' ' + token
            prev_is_word = generate.is_word(token)

            if token == '[MASK]':
                token = ''
            elif token == '[CLS]':
                token = '\n\n'
            elif token == '[SEP]':
                token = '\n'
            text += token
            yield urllib.parse.quote(token)
 
        generated += 1 
        app.logger.info("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n")
        app.logger.info(text)
    yield "[ALL SAMPLES GENERATED]"


class ReusableForm(Form):
    """User entry form for entering specifics for generation"""
    # Starting seed
    seed = TextField("請輸入一個起始句子:", default="豬肉", validators=[validators.InputRequired()])
  
    # Configure GPT2
    length = IntegerField("生成長度 (<=1024):", default=200, validators=[validators.InputRequired(), validators.NumberRange(-1, 1024)])
    temperature = DecimalField("文章生成的隨機度 (0.1-3):", default=1.2, places=1, validators=[validators.InputRequired(), validators.NumberRange(0.1, 3)])
    topk = IntegerField("前k大的機率頻繁詞抽樣:", default=50, validators=[validators.InputRequired()])
    topp = DecimalField("前k大的累積機率頻繁詞抽樣:", default=0, validators=[validators.InputRequired()])
    fast_pattern = BooleanField("采用更加快的方式生成文本:", default=False)
    nsamples = IntegerRangeField('生成幾個樣本:', default=2)
    modelname = SelectField('模型', validators=[validators.InputRequired()])
    fast_pattern = BooleanField("采用更加快的方式生成文本:", default=False)

    # Submit button
    submit = SubmitField("開始產生文章")

# Create app
app = Flask(__name__)
app.config['DEBUG'] = True

# Progress
@app.route("/progress", methods=['GET'])
def progress():
    print (request.args)
    # Extract information
    seed = request.args['seed']
    length = int(request.args['length'])
    temperature = float(request.args['temperature'])
    topk = int(request.args['topk'])
    topp = float(request.args['topp'])
    fast_pattern = 'fast_pattern' in request.args.keys()
    nsamples = int(request.args['nsamples'])
    modelname = request.args['modelname']

    def generate():
        # Generate a random sequence
        for x in text_generator(seed=seed,
                                     length=length,
                                     temperature=temperature,
                                     topk=topk,
                                     topp=topp,
                                     fast_pattern=fast_pattern,
                                     nsamples=nsamples,
                                     modelname=modelname):
            yield f"data:{x}\n\n"
    return Response(generate(), mimetype= 'text/event-stream')

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)

    # Determine model choices on every request so we don't need to
    # restart the server when adding or removing models
    form.modelname.choices = findModels()
    if len(form.modelname.choices) > 0:
        form.modelname.default = form.modelname.choices[0]

    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        return render_template('seeded.html', query=urllib.parse.urlencode(request.form))
    else:
        # Send template information to index.html
        return render_template('index.html', form=form)


if __name__ == "__main__":
    app.logger.info(("* Loading model and Flask starting server..."
                     "please wait until server has fully started"))
 
    # Run app
    app.run(host="0.0.0.0", port=8000)
