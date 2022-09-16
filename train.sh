sudo apt update
sudo apt install -y libgl1-mesa-glx
pip install timm -i https://mirrors.aliyun.com/pypi/simple/
pip install pyiqa -i https://mirrors.aliyun.com/pypi/simple/
pip install colour_demosaicing -i https://mirrors.aliyun.com/pypi/simple/
pip install matplotlib -i https://mirrors.aliyun.com/pypi/simple/
pip install thop -i https://mirrors.aliyun.com/pypi/simple/
pip install einops -i https://mirrors.aliyun.com/pypi/simple/
pip install pytorch_msssim -i https://mirrors.aliyun.com/pypi/simple/
pip install lpips -i https://mirrors.aliyun.com/pypi/simple/
pip install pyyaml -i https://mirrors.aliyun.com/pypi/simple/
pip install matplotlib -i https://mirrors.aliyun.com/pypi/simple/

python train.py --config configs/stdrunet.yml