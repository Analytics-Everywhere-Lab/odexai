# XAI Object Detection Review
## References
- D-CLOSE: https://github.com/Binh24399/D-CLOSE/
- G-CAME, D-RISE: https://github.com/khanhnguyenuet/GCAME

## Installation
- Install cmake
```
sudo apt-get install cmake
```
or
```
pip install cmake
```

- Install YOLOX
```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -U pip && pip install -r requirements.txt
pip install -v -e .
```

- Install requirements
```
pip install -r requirements.txt
```

## Notes
- D-CLOSE outputs saliency maps for all predicted boxes from models (YOLOX, FasterRCNN)
