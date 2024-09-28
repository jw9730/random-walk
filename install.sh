set -e
pip3 install numpy==1.26.4
pip3 install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip3 install xformers==0.0.24+cu118 --index-url https://download.pytorch.org/whl/cu118
pip3 install lightning==2.2.0.post0
pip3 install transformers==4.40.1
pip3 install causal-conv1d==1.2.0.post2 mamba-ssm==1.2.0.post1
pip3 install easydict==1.10 fastapi==0.100.1 wandb==0.15.8 plum-dispatch==2.1.1 scikit-learn==1.3.0 matplotlib==3.7.2 seaborn==0.12.2
pip3 install torch_geometric==2.5.0
pip3 install pyg_lib==0.4.0+pt22cu118 torch_scatter==2.1.2+pt22cu118 torch_sparse==0.6.18+pt22cu118 torch_cluster==1.6.3+pt22cu118 torch_spline_conv==1.2.2+pt22cu118 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip3 install tensorboard==2.13.0
pip3 install ogb==1.3.6
apt install -y ucommon-utils wget
pip3 install accelerate
pip3 install -r requirements.txt
pip3 install pybind11
bash install_walker.sh
