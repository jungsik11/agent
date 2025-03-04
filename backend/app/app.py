import sys , os, yaml
import uvicorn
from pathlib import Path

model_path = str(Path(os.path.abspath('')).parents[1]) + '/llm_model'
sys.path.append(model_path)

from fastapi import FastAPI
from run_llm import run_llm
from lora_train import lora_train


app = FastAPI()

conf_file = "/models/agent_conf.yaml"

@app.get("/model_list")
async def get_model_list():

    with open(model_path + conf_file,'r') as f:
        models = yaml.load(f, Loader=yaml.FullLoader)
    models = models['models']
    models = [i.split('/')[-1] for i in models]
    return {'model_list': models}


@app.get("/dataset_list")
async def get_model_list():

    with open(model_path + conf_file,'r') as f:
        datasets = yaml.load(f, Loader=yaml.FullLoader)
    datasets = datasets['datasets']
    datasets = [i.split('/')[-1] for i in datasets]
    return {'dataset_list': datasets}


@app.get("/generate")
async def generate(model, input_text):
    res = run_llm(model_path + '/models/' + model, input_text)
    return {"message": res}

@app.get("/train/lora")
async def train_lora(model, data, adapter):

    config_file = model_path + "/config/lora_config.yaml"
    adapter_path = model_path + '/adapters/' + model
    model = model_path + '/models/' + model
    dataset = model_path + '/data/' + data

    if os.path.exists(adapter_path)==False: os.mkdir(adapter_path)
    adapter_path += "/" + adapter

    if os.path.exists(adapter_path)==False: os.mkdir(adapter_path)
    res = lora_train(config_file, model, dataset,adapter_path)
    return {"message": res}



@app.get("/")
async def hello():
    return {"message": "Hello World"}





@app.get("/get_models")
def generate(input_text):

    return {"message": run_llm(input_text)}


if __name__ == "__main__" :
	uvicorn.run("app:app", reload=True)