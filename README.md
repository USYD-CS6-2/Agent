Here are Steps to deploy the agent module. We recommend you to set it up on a web server.

## Quick start

**Step 1: Clone the Repository**

First, download the project source code to your local machine or cloud server. Open your terminal and execute the following command to clone the repository:

```bash
git clone https://github.com/USYD-CS6-2/Agent.git
```

Once the download is complete, navigate into the root directory of the project:

```bash
cd Agent
```

**Step 2: Set Up a Virtual Environment**

It is highly recommended to create a virtual environment before installing any packages. This keeps the project dependencies isolated and prevents conflicts with your system's global Python environment.

Run the following commands to create and activate a virtual environment named .venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

(Note: If you are deploying on a Windows machine, use .venv\Scripts\activate to activate the environment.) 

**Step 3: Install Dependencies**

With your virtual environment activated, you can now install all the required Python libraries. We have listed all necessary packages (like FastAPI, LangGraph, and Pydantic) in the requirements file.

Run this command to install them:

```bash
pip install -r requirements.txt
```

**Step 4: Configure Environment Variables**

Open the .env file and update the API_KEY and BASE_URL with your own LLM API.

Update llm variable in agent_graph.py and batch_processor.py.

If you want to test the connection with LLM, or the effective of model:

```bash
run API_test_connection.py
```

```bash
run agent_graph.py
```

```bash
run batch_processor.py
```

(Note: *batch_processor.py* will uses default reddit json in the repository. You can check the main function and replace it.)

**Step 5: Start the Backend Server**

Once the environment is configured, you are ready to launch the backend application. We use uvicorn to run the FastAPI server.

Execute the following command to start the server and bind it to port 8001:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

If the deployment is successful, you will see a terminal message saying Uvicorn running on http://0.0.0.0:8001. The backend is now fully operational and ready to receive JSON requests from the Chrome extension. 
