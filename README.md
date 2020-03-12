# roLabelImg for PointClouds: a 2D top-down annotator for pointclouds with AI

## Features
This annotator tool loads pointclouds and automatically partially annotates them with a neural network. First the floor is detected to generate the top-down view and then a bed is located on the image and annotated with a bounding box (=*bbox*) on the top-down (central) view.

## Install instructions
Before installing this you should follow the install instructions on [this repository](https://github.com/vanderschuea/KapNet). This code will only run with `python3.6+`!!

### Ubuntu
`sudo apt-get install pyqt5-dev-tools`
`sudo pip3 install 'pyautogui' 'PyQt5'`
`make qt5py3`
`python3 labelImg.py`

### Windows (not tested yet)
Install Qt5 [here](https://www.riverbankcomputing.com/software/pyqt/download5)
`pip3 install 'pyautogui' 'PyQt5'`
`pyrcc5 -o libs/resources.py resources.qrc`
`python3 labelImg.py`

## Key- and Mouse-bindings
### Shortcuts
| Shortcut | Function |
|---|---|
|`Ctrl + u`   | Load all of the images from a directory    |
|`Ctrl + r`   | Change the default annotation target dir   |
|`Ctrl + s`   | Save annotation file  |
|`Alt  + s`   | Toggle autosave: when changing images with keyboard-shortcuts, the current one is automatically saved (*On* by default) |
|`Ctrl + d`   | Duplicate the current label and rect box   |
|`Ctrl + c`   | Copy all selected label and rect boxes     |
|`Ctrl + v`   | Paste & append all copied labels           |
|`Ctrl + shift + v`   | Paste all copied labels erasing all current labels |
|`w`          | Create a rect box                          |
|`e`          | Create a Rotated rect box                  |
|`d`          | Next image                                 |
|`q`          | Previous image                             |
|`r`          | Hidden/Show Rotated Rect boxes             |
|`n`          | Hidden/Show Normal Rect boxes              |
|`f`          | Hide/Show annotated floor pixels in segmented img |
|`del`        | Delete the selected rect box               |
|`Ctrl + +`    | Zoom in                                    |
|`Ctrl + -`    | Zoom out                                   |
|`↑/→/↓/←` (or `z/d/s/q`)| Keyboard arrows (or zdsq) to move selected rect box  |
|`z/x/c/v`     | Keyboard to rotate selected rect box       |

### Mouse
With the ***left click*** you can select a *bbox* to **move it around** or select a *corner* to **stretch** the *bbox*.
With the ***right click*** you can select a *shape* to **check its options** or select a *corner* to **rotate** the *bbox*.
Holding `Ctrl` and using the `ScrollWheel` you can zoom in or out.

## Known bugs
Sometimes the anottator will be 'stuck' with only `loading` displayed on the screen. If this goes on for more than 2-3 seconds press `Ctrl+Shift+R`. This bug is only visual and stems from a bad refresh of the canvas after a threaded operation.

## Licence
Free software: [MIT license](https://github.com/vanderschuea/roLabelImg/blob/master/LICENSE)
