# Edge Impulse Example: stand-alone inferencing (C++)

This builds and runs an exported impulse locally on your machine. See the documentation at [Running your impulse locally](https://docs.edgeimpulse.com/docs/running-your-impulse-locally). There is also a [C version](https://github.com/edgeimpulse/example-standalone-inferencing-c) of this application.

## Imagine demo

* Create an impulse with 320x320 image input, and a DSP block that converts to grayscale. No need to train the model.
* Create a file called `features.txt` and paste the 'Processed features' from an image into the file.
* Build and run the application:

    ```
    rm -f source/*.o && make -j && ./build/edge-impulse-standalone features.txt
    ```

* You now have a file called `debug.bmp` with the original image and any blocks that it found.
