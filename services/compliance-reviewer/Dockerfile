FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/app-root/src

COPY services/compliance-reviewer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /opt/app-root/src/shared/
COPY services/compliance-reviewer/agent.py .
COPY services/compliance-reviewer/agent_executor.py .
COPY services/compliance-reviewer/__main__.py .

EXPOSE 8003

CMD ["python", "__main__.py"]
