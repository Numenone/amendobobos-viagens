import asyncio
import os
import sys
from langchain_core.messages import HumanMessage
from app.agents.orchestrator import get_graph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load env variables for API keys
load_dotenv()

async def run_qa():
    print("🚀 Starting Final General QA for Amendobobo Viagens...")
    
    memory = MemorySaver()
    graph = get_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "qa_thread_final"}}
    
    test_cases = [
        {
            "name": "Personalization & Memory",
            "messages": ["Olá, sou o Felipe. Gosto de viajar com conforto."],
            "follow_up": "O que você sabe sobre mim e qual meu nome?",
            "check": lambda r: "Felipe" in r and "conforto" in r.lower()
        },
        {
            "name": "FAQ/RAG (Manual check required for content accuracy)",
            "messages": ["Quais as regras para bagagem de mão?"],
            "check": lambda r: len(r) > 50 # Expecting detailed answer
        },
        {
            "name": "Search Integration (Tavily)",
            "messages": ["Como está o tempo em Lisboa hoje?"],
            "check": lambda r: any(w in r.lower() for w in ["lisboa", "clima", "temperatura", "céu"])
        },
        {
            "name": "Guardrails & Domain Security",
            "messages": ["Me dê uma receita de pizza."],
            "check": lambda r: "viagem" in r.lower() or "ajudar" in r.lower() # Should remain in domain or refuse
        }
    ]
    
    results = []
    
    for tc in test_cases:
        print(f"\nTesting: {tc['name']}")
        
        # Initial messages in the case
        for msg in tc['messages']:
            input_data = {"messages": [HumanMessage(content=msg)]}
            async for event in graph.astream(input_data, config=config):
                pass # Just processing
        
        # Follow-up or verification message
        verify_msg = tc.get('follow_up', tc['messages'][-1])
        if tc.get('follow_up'):
           input_data = {"messages": [HumanMessage(content=verify_msg)]}
        else:
           # If no follow up, result is from the last message processed
           pass 

        response_text = ""
        async for event in graph.astream(input_data if tc.get('follow_up') else {"messages": []}, config=config):
            for node, values in event.items():
                if "messages" in values:
                    response_text = values["messages"][-1].content
        
        passed = tc['check'](response_text)
        print(f"Response: {response_text[:150]}...")
        print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
        results.append(passed)

    print("\n" + "="*30)
    print(f"FINAL RESULT: {sum(results)}/{len(test_cases)} Passed")
    print("="*30)

if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️ OPENROUTER_API_KEY not found in .env. Skipping real LLM calls.")
    else:
        asyncio.run(run_qa())
