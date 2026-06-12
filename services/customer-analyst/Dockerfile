FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/app-root/src

COPY services/customer-analyst/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /opt/app-root/src/shared/
COPY services/customer-analyst/agent.py .
COPY services/customer-analyst/agent_executor.py .
COPY services/customer-analyst/__main__.py .

EXPOSE 8001

CMD ["python", "__main__.py"]
