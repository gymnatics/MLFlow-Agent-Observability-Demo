FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/app-root/src

COPY services/risk-assessor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /opt/app-root/src/shared/
COPY services/risk-assessor/agent.py .
COPY services/risk-assessor/agent_executor.py .
COPY services/risk-assessor/__main__.py .

EXPOSE 8002

CMD ["python", "__main__.py"]
