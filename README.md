# Fast-tunerFM: An Efficient Vision-Language Fine-tuning Scheme for Medical Foundation Models

## SOTA Fine-tuned weights
If you just want the retinal FM vision weights resulting from our fine-tuning scheme:  
Fine-tuned weights for RETFound: https://drive.google.com/file/d/1XEkemQqEZMf_ayJPS9udlkQAzCL2HRsf/view?usp=sharing    
Fine-tuned weights for VisionFM: https://drive.google.com/file/d/1KPxULUnhiU0IrlVuA1vy687niSkJNJdz/view?usp=sharing   


## Fine-tuning

Navigate into Fast-tunerFM/

Create a new virtual environment in Fast-tunerFM/ and install requirements.txt

Text encoder weights: Download BERT weights here and put them under Fast-tunerFM/pretrained_weights/: https://drive.google.com/file/d/1_yvgtR5ZcWxJbMpWn4v2_Tgg4TI4d5oh/view?usp=sharing  

Vision encoder weights: Put your vision model in Fast-tunerFM/  

Our in-house training data is private so you will need to use your own. Edit Fast-tunerFM/ImageCaptionDataset.py accordingly. __getitem__ should return a list consisting of two elements: an image (torch tensor) and a report (string).

Then in the command line run:
```sh
python train.py --model_weights path/to/model
```

Once your model is trained, run the following script to extract the vision backbone. This will save it under linear_probing/_weights. Note this has only been tested on RETFound, VisionFM, Uni4Eye++, and our in-house MAE. You may need to alter it for another FM.
```sh
python get_vision_backbone_for_linprobing.py --path_to_model models/<model name>
```


## Downstream classification datasets
Duke iAMD: https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm  
Harvard Glaucoma: https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GDP  
Noor Eye Hospital: https://hrabbani.site123.me/available-datasets/dataset-for-oct-classification-50-normal-48-amd-50-dme  
OCTDL: https://data.mendeley.com/datasets/sncdhf53xc/4  
OCTID: https://borealisdata.ca/dataverse/OCTID  
NEHUT: https://data.mendeley.com/datasets/8kt969dhx6/1
