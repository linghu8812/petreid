# process data
# pip install -r requirements.txt
# python setup.py develop
cd ../data
unzip pet_biometric_challenge_2022.zip
unzip test.zip
mv test pet_biometric_challenge_2022
cd ../train
python split_dataset.py

# train data
cd tools
python source_pretrain/main.py  source_pretrain/config_swin_base_test.yaml --work-dir swin_base
python source_pretrain/main.py  source_pretrain/config_swin_large_test.yaml --work-dir swin_large

# predict
cd ..
python predict_ensamble.py --resume ./logs/swin_base/checkpoint.pth ./logs/swin_large/checkpoint.pth \
    --config ./logs/swin_base/config.yaml ./logs/swin_large/config.yaml
    