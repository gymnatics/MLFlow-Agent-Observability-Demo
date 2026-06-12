FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/app-root/src

COPY services/orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /opt/app-root/src/shared/
COPY services/orchestrator/agent.py .
COPY services/orchestrator/agent_executor.py .
COPY services/orchestrator/__main__.py .

EXPOSE 8000

CMD ["python", "__main__.py"]
