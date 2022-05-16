python3 -m venv ingenious-env
source ingenious-env/bin/activate
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install git+https://github.com/huggingface/accelerate
pip3 install -r requirements.txt
git clone https://github.com/decile-team/submodlib.git
cd submodlib
pip3 install .
cd ..
deactivate