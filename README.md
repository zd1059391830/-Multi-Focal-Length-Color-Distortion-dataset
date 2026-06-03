# Multi-Focal-Length Color Mapping Demo

This repository provides a demo executable file and a small subset of publicly available data for functional verification of the color mapping process under multi-focal-length imaging.

The uploaded demo is intended for testing the basic workflow, including loading paired 1x and 3x images, automatically matching the corresponding image pair, and visualizing the color-mapped result. Only part of the publicly usable data is provided in this repository for demonstration purposes.

## Files

Please download the following files/folders:

text
demo.exe
Data/

The Data folder contains example image pairs organized by focal length. The 1x images and 3x images should have the same file names and be stored in corresponding folders, for example:
Data/Cutting/rewhite/test/1x/2.png
Data/Cutting/rewhite/test/3x/2.png

## How to Use
**1. **Download the demo executable file and the Data folder.
**2. **Double-click the executable file to launch the demo program.
**3. **In the upper-left corner of the interface, click the "打开1x文件夹" button.
**4. **Select the 1x folder. Please note that you should select the folder itself, not an individual image file.
**5. **After the folder is loaded, a list of image names will appear on the left side of the interface.
**6. **Double-click an image name in the left list.
**7. **The right side of the interface will display three images:
      * Cropped 1x image
      * 3x image
      * Color-mapped 3x image

These results can be used to visually verify whether the demo program correctly loads the paired images and performs the color mapping process.

## Data Organization

The 1x and 3x images are matched by file name. For example, if the selected 1x image is:

Data/Cutting/rewhite/test/1x/2.png

the program will automatically search for the corresponding 3x image:

Data/Cutting/rewhite/test/3x/2.png

If the corresponding 3x image is not found, the program will show a missing-image message.

## Purpose

This demo is provided only for functional verification and demonstration. The full dataset and complete experimental resources are not included in this repository.

## Notes
The demo executable is provided for quick testing.
The uploaded data are only a partial subset of the publicly usable data.
The program does not require users to manually select paired 3x images. The corresponding 3x image is automatically located according to the selected 1x image name.
Please keep the relative folder structure of the Data directory unchanged when testing the demo.
