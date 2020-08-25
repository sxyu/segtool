## GrabCut GUI tool
- A simple OpenCV interactive GrabCut GUI, partly adapted from
 https://docs.opencv.org/3.4/d8/d34/samples_2cpp_2grabcut_8cpp-example.html
- Usage `segtool <image_path>`
    - Expects a mask `<image_path>_mask.png` (1-channel grayscale image, 0 = background, 255 = foreground)
        - You can obtain this using e.g. included PointRend (see `pointrend` folder)
    - Repeat:
        - Left click to paint foreground and right click to paint BG; shift+click to erase brushstroke
        - After doing some painting, press space to run GrabCut and update the mask
        - Press 1-4 for different image/mask visualizations
        - Middle click+drag to pan, Ctrl+middle click+drag to zoom, Shift+middle click+drag to change brush size
    - When satisfied, press `S` to overwrite the mask image with GrabCut result
        - Backs up original mask to `<image_path>_mask_orig.png` and also saves `<image_path>_mask_gc.png` which allows GrabCut to resume state
          (used if editing same image again after closing)
    - Press `ESC` to exit

<img src="https://github.com/sxyu/segtool/blob/master/readme-img/grabcut-short.gif"
    width="400">

### Install
#### Dependencies
- Build tools:
    - C++ Compiler (gcc or msvc; C++11 support required)
    - CMake 
- OpenCV 3 or 4 (note: even if you've been using cv2 in Python you may not have this installed on the system)
    - Ubuntu: `sudo apt install libopencv-dev`
    - Mac: try using homebrew `brew install opencv`, if it doesn't work you'll have tobuild from source
    - Windows: download installer from https://opencv.org/releases/

#### Instructions
On Unix
`mkdir build && cd build && cmake .. && make -j12`.

Consider using `sudo make install` to install the program.

On Windows you can install Visual Studio and use
`mkdir build && cd build && cmake .. && cmake --build . --config Release`.

#### Dependencies
- OpenCV 3+


### GUI Controls

- ESC: quit

- Space: run GrabCut iteration, after painting some FG/BG
- LEFT CLICK: Paint FG
- RIGHT CLICK: Paint BG
- CTRL+LEFT CLICK: Paint 'maybe FG'
- CTRL+RIGHT CLICK: Paint 'maybe BG'
- SHIFT+LEFT CLICK: Erase brushstroke

- S: Save mask to overwrite original mask image (original backed up to 
*_mask_orig.png)
- R: Reset GrabCut
- C: Reset zoom/pan
- U: Clear current active brushstrokes (basic undo)
- =/-: Increase/decrease brush size
- 1-4: Blend/image/masked image/binary mask views
- MIDDLE CLICK + Drag: Pan
- CTRL+MIDDLE CLICK + Drag vertically: Zoom
- SHIFT+MIDDLE CLICK + Drag vertically: Change brush size
