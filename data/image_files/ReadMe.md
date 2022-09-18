Add your image frame files recorded using realsense camera here.
Recomended directory structure is as follows.

Directory name format : `R6_G4_L_14_9`
Discription: Room 6, Gutter 4, left side, 14th of 9th month. any other optional labels can be appended to the name.
folders containing different images can be added under this forder. An example is shown below:

```
data
    |
    bag_files
        ...
    image_files
        |
        R6_G4_L_14_9
            |
            color
                |
                00000.jpg
                00001.jpg
                ...
            depth
                |
                00000.png
                00001.png
                ...
            thermal
                |
                00000.png
                00001.png
                ...
            ...
            camera_intrinsic.json
```

