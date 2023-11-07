""" 
Prepare data for LayoutLM model. 
Input: the path to the data directory.
Output: A Dataset object containing the data.
"""
import json
import sys
sys.path[0]=''

from code.data_preparation.data_utils import create_dataset, load_data_text_and_boxes, load_data_text_only, load_data_text_and_boxes_and_images

# Load dataset-specific configuration
def load_dataset_with_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    data_dir = config['data_dir']
    input_format = config['input_format']
    preprocessing_config = config.get('preprocessing', {})
    
    if input_format == "text_and_boxes_and_images":
        # Load data with text and images
        data = load_data_text_and_boxes_and_images(data_dir)
    elif input_format == "text_and_boxes":
        # Load data with text and boxes
        data = load_data_text_and_boxes(data_dir)
    else: # input_format == "text_only"
        # Load data in text-only format
        data = load_data_text_only(data_dir)

dataset = load_dataset_with_config('config/funsd_config.json')
print(dataset)