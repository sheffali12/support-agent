@echo off
docker run -p 8000:8000 --env-file .env support-agent
start http://localhost:8000