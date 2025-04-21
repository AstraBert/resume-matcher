eval "$(conda shell.bash hook)"

conda activate resume-matcher
cd /server/
python3 server.py
conda deactivate
