FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/app-root/src

COPY services/mongodb-mcp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /opt/app-root/src/shared/
COPY services/mongodb-mcp/server.py .
COPY services/mongodb-mcp/seed_data.py .

EXPOSE 8090

CMD ["python", "server.py", "--port", "8090"]
