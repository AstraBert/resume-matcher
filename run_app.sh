eval "$(conda shell.bash hook)"

conda activate resume-matcher
cd /app/
uvicorn api:app --host 0.0.0.0 --port 80
conda deactivate
