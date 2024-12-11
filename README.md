- Chạy trên Google Colab

- Clone code về default folder trên google drive.
```
!git clone https://github.com/vanducngo/SinSR.git
```

- Checkout về target branch
```
%cd /content/SinSR
!git pull
# !git checkout default_drive
# !git checkout default_drive_light
!git checkout single_step
```

- Install requirement
```
!pip install -r /content/SinSR/requirements.txt
```

- Chạy chương trình 
```
!python /content/SinSR/app.py --colab
```