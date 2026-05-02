from fastapi import FastAPI
from pydantic import BaseModel
from db import init_db, insert_result, get_leaderboard

app = FastAPI()

class ResultIn(BaseModel):
    run_id: str
    model: str
    category: str
    prompt_id: str
    rule_score: float
    llm_score: float
    ensemble_score: float

@app.on_event("startup")
def startup():
    init_db()

@app.post("/results")
def post_result(res: ResultIn):
    insert_result(
        res.run_id, res.model, res.category, res.prompt_id, 
        res.rule_score, res.llm_score, res.ensemble_score
    )
    return {"status": "ok"}


@app.get("/leaderboard")
def leaderboard():
    return get_leaderboard()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)