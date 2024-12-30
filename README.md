# Cricket: A Self-Powered Chirping Pixel
### [[Paper]](https://cave.cs.columbia.edu/Statics/publications/pdfs/Nayar_TOG24.pdf) [[Project Page]](https://cave.cs.columbia.edu/projects/categories/project?cid=Computational+Imaging&pid=Cricket+A+Self-Powered+Chirping+Pixel) [[Video]](https://cave.cs.columbia.edu/old/projects/cricket/videos/TOG24_cricket_compressed.mp4)

[Shree K. Nayar](https://www.cs.columbia.edu/~nayar/), [Jeremy Klotz](https://cs.columbia.edu/~jklotz), Nikhil Nanda, and Mikhail Fridberg

This repository contains the code, schematic, and PCB layout for the paper
"Cricket: A Self-Powered Chirping Pixel" at SIGGRAPH 2024.


## Signal Processing
Run `python signal_processing/peak_detection.py` to execute the chirp detection
routine on raw data captured from a software-defined radio. The data file 
(`data.bin`) contains the IQ samples captured by a bladeRF radio, tuned
to 2055 MHz with a 50 MHz sampling rate.

## Circuit Schematic and Layout
The schematic and PCB layout of the cricket prototype are stored as an 
Altium Designer project in the zip file `hardware/altium-project.zip`. 
We have also included a PDF of the schematic in `hardware/schematic.pdf`. 

Note that the voltage regulator U4 is unused in our prototype; we have bypassed
that component with a jumper.
The circket's carrier frequency is determined by the voltage `VFREQ`.
The carrier frequency is set by modifying the resistors from `VFREQ` to `GND` 
until the desired frequency is attained.

## Citation
```
@article{nayar2024cricket,
    author = {Nayar, Shree K. and Klotz, Jeremy and Nanda, Nikhil and Fridberg, Mikhail},
    journal = {ACM Transactions on Graphics / SIGGRAPH},
    number = {4},
    volume = {43},
    pages = {151:1-11},
    title = {Cricket: A Self-Powered Chirping Pixel},
    month = {July},
    year = {2024},
}
```
