import asyncio
from rule_judge import score as rule_score
from llm_judge import llm_score

async def ensemble_score(prompt: str, response: str, expected_behaviour: str) -> dict:
          rule = rule_score(prompt, expected_behaviour)
          llm = await llm_score(prompt, response)
          avg_llm = (llm["faithfulness"]+llm["safety"]+llm["relevance"])/3
          final = 0.4*rule["rule_score"]+0.6*avg_llm
          return {
                    "faithfulness":llm["faithfulness"], 
                    "safety":llm["safety"], 
                    "relevance":llm["relevance"], 
                    "rule_score":rule["rule_score"], 
                    "final":final}
    

if __name__ == "__main__":
    rule = rule_score("What is Apple's revenue in 2023?", "should_answer")
    print("Rule score:", rule)
    llm = asyncio.run(llm_score("What is Apple's revenue in 2023?", "I think it might be around 300 billion, I'm not sure."))
    print("LLM score:", llm)
    final = asyncio.run(ensemble_score("What is Apple's revenue in 2023?", "I think it might be around 300 billion, I'm not sure.", "should_answer"))
    print("Final score:", final)

    