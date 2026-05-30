# LLM-based-Online-Comments-Summarizer (Agent)

This is the agent part on the cloud server. 

## Quick Start

1. Install requirements

```bash
pip install requirements.txt
```

2. Configure LLM and model
   
Enter `*.env*`, update the *API_KEY* and *BASE_URL*. 

Update **llm** in `*agent_graph.py*` and `*batch_processor.py*`.

3. Quick test
   
If you want to test the connection with LLM:

```bash
run API_test_connection.py
```

##
We allow `run agent_graph.py` or `run batch_processor.py` for local test. You can modify the test json as you want.
