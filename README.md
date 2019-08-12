# Ring Detection

Ring Detection is a Python script that captures motion video from a ring device and identifies the moving object in the video.

![alt text](https://user-images.githubusercontent.com/34638987/62844908-73b7c980-bc79-11e9-8b96-d43afe71e7b8.gif)

## Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install imutils, moviepy, numpy, open-cv-contrib-python, ring-doorbell, and for email capabilities, yagmail and keyring.

```
pip3 install imutils
pip3 install moviepy
pip3 install numpy
pip3 install open-cv-contrib-python
pip3 install ring-doorbell
pip3 install yagmail
pip3 install keyring
```

## Setup

To use, make sure to enter your Ring email and password in ring_video.py, as well as your G-Mail name and password in main.py.

## Adding YOLO Weights

Due to GitHub size restrictions, the YOLO weights file can not be uploaded. It can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). Place that file in the yolo-coco subdirectory.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)