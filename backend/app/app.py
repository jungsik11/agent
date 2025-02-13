
from fastapi import FastAPI
from run_llm import run_llm
import uvicorn
app = FastAPI()


@app.get("/")
async def hello():
    return {"message": "Hello World"}



@app.get("/generate")
async def generate(input_text):

    return {"message": run_llm(input_text)}


if __name__ == "__main__" :
	uvicorn.run("app:app", reload=True)