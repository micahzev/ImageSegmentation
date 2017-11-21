# Pixel Segmentation

This collection of Python scripts runs incisor segmentation on dental radiographs for the four upper and lower incisors.

The starting point and main file is named `segment_main.py`.

This controls the flow and usage of all the auxiliary code that is used by the program.

There are two settings. Automatic and Manual. You can run the segmentation using both.

You can also suggest how many incisors to segment.

Currently the setup allows for a leave-one-out analysis, you can choose the value of `x`, which is the radiograph to leave out in the run.

F-Measures are logged per incisor.

***

License
----

MIT