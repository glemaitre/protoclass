Protoclass module
=========

#### Manifesto

Because *Human* is **perfectible** and **error-prone**, because *Science* should be **open** and **flow** and because *cogito ergo sum*.

Goal
----

This toolbox will aggregate some machine learning tools from all the walk of lives in order to fast prototype.

File Structure
--------------

    protoclass
    |- classification/						
    |  +- UnbalancedDataset/					
    |  +- test/							
    |  |- __init__.py						
    |  |- classification.py					
    |
    |- extraction/						
    |  +- test/							
    |  |- __init__.py						
    |  |- edge_analysis.py					
    |  |- sampling.py						
    |  |- texture_analysis.py					
    |
    |- pipeline/						
    |  |- feature-classification/				
    |  |  |- classification_haralick.py				
    |  |
    |  |- feature-detection/					
    |  |  |- detection_edge_rician.py				
    |  |  |- detection_haralick_gnormalised.py			
    |  |  |- detection_haralick_rnormalised.py			
    |  |  |- detection_haralick_unormalised_cluster.py		
    |  |
    |  |- feature-normalisation/				
    |  |  |- normalisation_t2w_cluster.py			
    |  |
    |  |- feature-sampling/					
    |  |  |- sampling_haralick_data.py				
    |  |
    |  |- feature-validation/					
    |  |  |- validation_classification_haralick.py		
    |
    |- preprocessing/						
    |  +- test/							
    |  |- __init__.py						
    |  |- normalisation.py					
    |
    |- selection/						
    |  +- test/							
    |  |- __init__.py						
    |
    |- tool/							
    |  +- test/							
    |  |- __init__.py						
    |  |- dicom_manip.py					
    |
    |- validation/						
    |  +- test/								
    |  |- init.py						
    |  |- validation.py						
    |
    |- README							
    |- __init__.py						

Execution
---------

Check the folder `protoclass/pipeline`.
