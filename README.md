# Vocoder: HiFiGAN

## Get the code
```shell
git clone https://github.com/ainmukh/hw4_tts
cd hw4_tts
```

## Installation guide
```shell
pip install -q -r ./requirements.txt
```

## Testing
```shell
wget -q https://www.dropbox.com/sh/8qwut7xeec45uja/AABBfi86LmkjvQxnCmq86H2Ga
unzip AABBfi86LmkjvQxnCmq86H2Ga -d data
mkdir saved/
mv data/model_best.pth saved/model_best.pth
cp config1.json saved/config.json
python test.py -c config1.json
```
