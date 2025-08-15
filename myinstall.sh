# copilot generated script
# Not ensured to work

ml load miniconda3
conda create -n videograin python==3.10
conda activate videograin
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --pre -U xformers==0.0.27

pip install -r requirements.txt
pip install onnxruntime onnxruntime-gpu --force-reinstall
pip install --force-reinstall --no-deps scikit-learn
pip install --force-reinstall numpy