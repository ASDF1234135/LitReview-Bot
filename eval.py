import asyncio
import pandas as pd
from ragas import evaluate
from ragas.metrics import context_recall, faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datasets import Dataset

from agent_core import research_agent

judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

test_data = [
    {
        "question": "What is the core concept of Active Learning (AL)?",
        "ground_truth": "Active Learning is a subfield of machine learning based on the idea that if a small amount of valuable data can be selected from an unlabeled pool for annotation, a better-performing model can be obtained with less training cost.",
        "user_id": "default_user"
    },
    {
        "question": "What are the main components of a traditional Active Learning framework?",
        "ground_truth": "A traditional AL system consists of an unlabeled pool, a query strategy, an oracle, a labeled pool, an active learner, and a stopping strategy.",
        "user_id": "default_user"
    },
    {
        "question": "How does the Least Confidence uncertainty sampling strategy work?",
        "ground_truth": "Least Confidence selects the instance for which the model is least confident in its prediction, specifically finding the instance with the lowest probability for the predicted class.",
        "user_id": "default_user"
    },
    {
        "question": "What is the 'Query by Committee' (QBC) strategy in active learning?",
        "ground_truth": "QBC samples a set of hypotheses (committee) and measures the disagreement among them (often using vote entropy) to select instances where the committee members disagree the most.",
        "user_id": "default_user"
    },
    {
        "question": "How does this study define 'Generative Active Learning'?",
        "ground_truth": "Generative Active Learning is defined as an AI model learning method that incorporates any Large Language Models (LLMs) into the active learning process.",
        "user_id": "default_user"
    },
    {
        "question": "What are the two main roles an LLM can play in Generative Active Learning?",
        "ground_truth": "In Generative AL, an LLM can serve as a 'Query Strategy' (selecting or filtering samples) or as an 'Oracle' (annotating data).",
        "user_id": "default_user"
    },
    {
        "question": "What is the LDCAL method described in the literature review?",
        "ground_truth": "LDCAL prompts an LLM to partition the unlabeled pool into four difficulty levels (Easy, Moderate, Challenge, Hard) and then applies certainty gain maximization to sample equally from each for fine-tuning.",
        "user_id": "default_user"
    },
    {
        "question": "How does the 'FreeAL' method attempt to achieve human-free active learning?",
        "ground_truth": "FreeAL uses a strong model to label data and a weak model to distinguish noise labels, allowing for joint training where the strong model acts as the annotator without human intervention.",
        "user_id": "default_user"
    },
    {
        "question": "What is the main limitation of existing Generative AL methods identified in this proposal?",
        "ground_truth": "Most existing methods treat LLMs solely as query strategies or oracles without incorporating the real-time training dynamics (learning state) of the active learner into the decision-making process.",
        "user_id": "default_user"
    },
    {
        "question": "What is the 'Direct Feedback' mechanism proposed in this research?",
        "ground_truth": "Direct Feedback involves converting the training dynamics of the active learner into textual descriptions and integrating them directly into the LLM's prompt for query selection.",
        "user_id": "default_user"
    },
    {
        "question": "What is the 'Indirect Feedback' mechanism proposed in this research?",
        "ground_truth": "Indirect Feedback leverages the learning discrepancies between a strong model and a weak model to indirectly reflect the learner's state and guide the query strategy.",
        "user_id": "default_user"
    },
    {
        "question": "What research gap does this project aim to address?",
        "ground_truth": "The project aims to address the lack of a closed-loop mechanism that integrates the active learner's feedback (training dynamics) into LLM-based query decision-making.",
        "user_id": "default_user"
    },
    {
        "question": "How can LLMs help with the 'cold-start' problem in Active Learning?",
        "ground_truth": "LLMs can use their powerful few-shot reasoning capabilities to select informative samples or generate synthetic data in the initial phase, mitigating the difficulty of selecting samples when the model is untrained.",
        "user_id": "default_user"
    },
    {
        "question": "What is the difference between 'LLM Selection' and 'LLM Generation' as query strategies?",
        "ground_truth": "LLM Selection uses the LLM to filter, score, or directly select samples from a pool, whereas LLM Generation uses the LLM to generate answers or explanations (like Chain-of-Thought) to serve as the basis for selection.",
        "user_id": "default_user"
    },
    {
        "question": "According to the timeline, when is the evaluation phase scheduled?",
        "ground_truth": "The evaluation phase is scheduled for March 2026 to April 2026.",
        "user_id": "default_user"
    }
]

async def run_agent_for_eval():
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print("--- Starting Evaluation Run ---")
    
    for item in test_data:
        print(f"Testing: {item['question']}")
        
        initial_state = {
            "question": item["question"],
            "user_id": item["user_id"],
            "router_decision": "",
            "local_contexts": [],
            "external_contexts": [],
            "external_docs": [],
            "is_sufficient": False,
            "sources": [],
            "final_answer": "",
            "search_history": [],
            "retry_count": 0
        }
        
        output = await research_agent.ainvoke(initial_state)
        
        results["question"].append(item["question"])
        results["answer"].append(output.get("final_answer", ""))

        all_ctx = output.get("local_contexts", []) + output.get("external_contexts", [])
        results["contexts"].append(all_ctx)
        results["ground_truth"].append(item["ground_truth"])

    return results

def main():
    data_dict = asyncio.run(run_agent_for_eval())
    
    dataset = Dataset.from_dict(data_dict)
    
    print("--- Running Ragas Metrics (LLM as a Judge) ---")
    scores = evaluate(
        dataset=dataset,
        metrics=[context_recall, faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=embeddings
    )
    
    df = scores.to_pandas()
    df.to_csv("eval_results_baseline.csv", index=False)
    col_question = "user_input" if "user_input" in df.columns else "question"
    
    target_cols = [col_question, "context_recall", "faithfulness", "answer_relevancy"]
    final_cols = [c for c in target_cols if c in df.columns]

    print("\n=== Evaluation Report ===")
    print(df[final_cols])
    
    print("\n=== Average Scores ===")
    print(scores)
    

if __name__ == "__main__":
    main()