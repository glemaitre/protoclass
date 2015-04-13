EMBC 2015
=========

This is a study of MRI data alignment in Prostate imaging.
Targeting EMBC 2015 conference

Thesis
------

The MRI data between patients is not aligned. Therefore normalizing, speriphing, etc. the data has makes no sense when the features are not aligned.

Goal
----

* Illustrate the problem
* Propose an alignment method
* try to calculate xxxxxx

File Structure
--------------

    project
    |- doc/             # documentation for the study
    |  +- paper/        # manuscript(s), whether generated or not
    |
    |- data             # raw and primary data, are not changed once created 
    |  |- raw/          # raw data, will not be altered
    |  +- clean/        # cleaned data, will not be altered once created
    |
    |- src/             # any programmatic code
    |  |- experiment-t2w.ipynb # main notebook for the T2 experiment
    |  |- README               # software structure description
    |
    |- results          # all output from workflows and analyses
    |  |- figures/      # graphs, likely designated for manuscript figures
    |  +- pictures/     # diagrams, images, and other non-graph graphics
    |
    |- scratch/         # temporary files that can be safely deleted or lost
    |- README           # the top level description of content

Execution
---------

```
cd ./src/
ipython notebook
```

Todo
----

- [ ] Add project information goals, etc... @guillaume
- [ ] change src/README.md
- [ ] Change data/README description @guillaume
- [ ] 

