# LLM-based-Online-Comments-Summarizer
This is the agent part on the cloud server. 

If you want to use another LLM, please update `*.env*`, and **llm** in `*agent_graph.py*` and `*batch_processor.py*`.

If you want to test the connection, please update **llm** in `*agent_graph.py*` and `*batch_processor.py*` and directly `run API_test_connection.py` in the terminal.

We allow `run agent_graph.py` or `run batch_processor.py` for local test. You can modify the test json as you want.