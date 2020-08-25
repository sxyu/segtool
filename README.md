## GrabCut GUI tool
- A simple OpenCV interactive GrabCut GUI, partly adapted from
 https://docs.opencv.org/3.4/d8/d34/samples_2cpp_2grabcut_8cpp-example.html
- Usage `segtool <image_path>`
    - Expects a mask `<image_path>_mask.png` (1-channel grayscale image, 0 = background, 255 = foreground)
        - You can obtain this using e.g. included PointRend (see `pointrend` folder)
    - Overwrites the mask image with GrabCut result (upon pressing `S`)
    - Backs up original mask to `<image_path>_mask_orig.png`

<img src="https://github.com/sxyu/segtool/blob/master/readme-img/grabcut-short.gif"
    width="400">

### Controls

- ESC: quit

- Space: run GrabCut iteration, after painting some FG/BG
- LEFT CLICK: Paint FG
- RIGHT CLICK: Paint BG
- CTRL+LEFT CLICK: Paint maybe FG
- CTRL+RIGHT CLICK: Paint maybe BG;

- S: Save mask to overwrite original mask image (original backed up to 
*_mask_orig.png)
- R: Reset GrabCut
- T: Clear current active strokes (does not modify actual mask)
- =/-: Increase/decrease brush size
- u/i/o/p: Blend/image/masked image/binary mask views
- MIDDLE CLICK + Drag: Pan
- CTRL+MIDDLE CLICK + Drag vertically: Zoom
