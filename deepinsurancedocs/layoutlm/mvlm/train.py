'''
Train the model on Masked Visual Language Modeling task, either from a pre-trained model either from a saved checkpoint.
'''
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime

import torch
import re 
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler  import LinearLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import AdamW, LayoutLMTokenizer, DataCollatorForLanguageModeling, LayoutLMConfig

sys.path[0] = ''  # nopep8
sys.path.append('data/docile')

from docile.dataset import CachingConfig, Dataset, Document

from deepinsurancedocs.data_preparation.data_utils import label_dict_transform  # nopep8
from deepinsurancedocs.data_preparation.layoutlm_docile_dataset import LayoutLMDocileDataset  # nopep8
from deepinsurancedocs.data_preparation.layoutlm_dataset import LayoutLMDataset  # nopep8
from deepinsurancedocs.layoutlm.token_classif.train import train_token_classifier  # nopep8
from models.layoutlm_mvlm.prepare_data import LayoutLMDataPreparationMVLM  # nopep8
from models.layoutlm_mvlm.train_utils import eval, train_epoch, layoutlm_collate_fn  # nopep8
from models.layoutlm_mvlm.model import LayoutLMForMaskedLMInternal


def main():
    # -------------------------------------------------------------------------------------------- #
    #                                         Training args                                        #
    # -------------------------------------------------------------------------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    config_path = args.config_path
    SAVE_MODEL_PATH = args.output_dir

    # Debug parameters
    # config_path = 'config/mvlm_docile_5k.json'

    model_name = "layoutlm_mvlm"
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # -------------------------------------------------------------------------------------------- #
    #                        Open Config with dataset & training information                       #
    # -------------------------------------------------------------------------------------------- #
    
    # config_path="/mnt/config/mvlm_docile_5k.json"
    # print(f"CAREFUL !!! DEBUG MODE USING {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        # config is a dict to store the following information about the dataset:
        # - data_dir: path to the directory containing the dataset
        # - input_format: format of the input data
        # - preprocessing: dictionary containing the tagging scheme used for preprocessing
        # - label_list: dictionary containing the mapping of labels to their corresponding indices
        config = json.load(f)
    DATA_DIR = config.get('data_dir', None)
    # IS_DOCILE = config.get('is_docile', False)
    VAL_DATA_DIR = config.get('validation_data_dir')
    VAL_DATA_NAME = VAL_DATA_DIR.split('/')[-1]
    assert VAL_DATA_NAME != ''
    DATASET_NAME = config.get('dataset_name', 'no_dataset_name') # delete .json extension
    PRETRAINED_MODEL = config.get('pretrained_model', '')
    CHECKPOINT_PATH = config.get('checkpoint_path', None)
    BATCH_SIZE = config['training_parameters'].get('batch_size', 1)
    LEARNING_RATE = config['training_parameters'].get('learning_rate', 0.001)
    NUM_TRAIN_EPOCHS = config['training_parameters'].get('epoch_num', 1)
    LOCAL_FILES_ONLY = config.get('is_local_model', False)
    ACCUMULATION_STEPS = config['training_parameters'].get('gradient_accumulation_steps', 1)
    TAGGING_SCHEME = config['preprocessing'].get('tagging_scheme', None)
    MODEL_DIR = config.get('model_dir', None)
    
    # Open os.path.join(DATA_DIR, 'label_list.json')
    with open(os.path.join(DATA_DIR, 'label_list.json'), 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    subset_index_path = os.path.join(DATA_DIR, DATASET_NAME,'list_pages.json')

    print(config['training_parameters'])

    # -------------------------------------------------------------------------------------------- #
    #                                           Tokenizer                                          #
    # -------------------------------------------------------------------------------------------- #
    tokenizer = LayoutLMTokenizer.from_pretrained(os.path.join(MODEL_DIR, "microsoft/layoutlm-base-uncased"))
    idx2label = label_dict_transform(label_dict=label_dict, 
                                     scheme=config['preprocessing']['tagging_scheme'])
    label2idx = {label: idx for idx, label in idx2label.items()}
    label_list = list(idx2label.values())
    num_labels = len(label_list)

    # -------------------------------------------------------------------------------------------- #
    #                                             Model                                            #
    # -------------------------------------------------------------------------------------------- #
    if PRETRAINED_MODEL:
        model = LayoutLMForMaskedLMInternal.from_pretrained(PRETRAINED_MODEL)
    else:
        default_config=LayoutLMConfig()
        model = LayoutLMForMaskedLMInternal(default_config)

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    best_epoch = NUM_TRAIN_EPOCHS
    global_step = 0

    # -------------------------------------------------------------------------------------------- #
    #                                             Data                                             #
    # -------------------------------------------------------------------------------------------- #
    print('Loading Docile Unlabeled Dataset...')

    print('Loading MVLM Train Dataset...')
    train_dataset = LayoutLMDocileDataset(DATA_DIR, # Path to the index of the subset we are using
                                          subset_index_path,
                                          tokenizer, 
                                          label_list, 
                                          pad_token_label_id, 
                                          'train',
                                          TAGGING_SCHEME)

    print('Loading MVLM Validation Dataset...')
    val_dataset = LayoutLMDocileDataset(DATA_DIR,
                                        os.path.join(DATA_DIR, 'list_data_train.json'),
                                        tokenizer,
                                        label_list,
                                        pad_token_label_id,
                                        'test',
                                        TAGGING_SCHEME)


    data_collator = DataCollatorForLanguageModeling(tokenizer, 
                                                    mlm=True, 
                                                    mlm_probability=0.15)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=data_collator,
                                  num_workers=6 # Nb of workers to modulate depending on CPU
                                  )

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,
                                 sampler=val_sampler,
                                 batch_size=BATCH_SIZE,
                                 collate_fn=data_collator,
                                 num_workers=6 # Nb of workers to modulate depending on CPU
                                 )
    # -------------------------------------------------------------------------------------------- #
    #                                            Logging                                           #
    # -------------------------------------------------------------------------------------------- #
    if not os.path.exists(SAVE_MODEL_PATH + '/logs/'):
        os.makedirs(SAVE_MODEL_PATH + '/logs/')
    log_file = f'train_{model_name}_{DATASET_NAME}.log'
    logging.basicConfig(filename=SAVE_MODEL_PATH + '/logs/' + log_file,
                        level=logging.INFO)
    logging.info('Training settings')
    logging.info(args)

    writer_train = SummaryWriter(log_dir=f"{SAVE_MODEL_PATH}/mvlm_runs/mvlm/train")
    writer_val = SummaryWriter(log_dir=f"{SAVE_MODEL_PATH}/mvlm_runs/mvlm/val")
    writer_test_tc = SummaryWriter(log_dir=f"{SAVE_MODEL_PATH}/mvlm_runs/mvlm/test_tc")
    writers = {'train': writer_train, 'val': writer_val, 'test_tc': writer_test_tc}
    
    # csv_data saves the data of MVLM Train, MVLM Val & TC eval during MVLM (to see how TC reuslts evolve with MVLM)
    csv_data = {'train': {'step': [],
                          'loss': [],
                          'accuracy': [], 
                          'perplexity': []}, 
                'val': {'step': [],
                          'loss': [],
                        'accuracy': [], 
                        'perplexity': []}, 
                'test_tc': {'step': [],
                            'loss': [],
                            'f1': [], 
                            'doc_exact_match': []}}

    # -------------------------------------------------------------------------------------------- #
    #                                           Training                                           #
    # -------------------------------------------------------------------------------------------- #
    # ---------------------------------------- First Eval ---------------------------------------- #
    model.eval()
    with torch.no_grad():
        val_results = eval(model, 
                           tokenizer,
                           val_dataloader, 
                           device, 
                           pad_token_label_id,
                           logging=logging, 
                           print_results=True)

        csv_data['val']['step'].append(global_step)
        csv_data['val']['loss'].append(float(val_results["loss"]))
        csv_data['val']['accuracy'].append(float(val_results["accuracy"]))
        csv_data['val']['perplexity'].append(float(val_results["perplexity"]))
        writer_val.add_scalar("Loss/mvlm", val_results["loss"], global_step=global_step)
        writer_val.add_scalar("Metrics/accuracy", val_results["accuracy"], global_step=global_step)
        writer_val.add_scalar("Metrics/perplexity",val_results["perplexity"], global_step=global_step)

    # ---------------------------------------- Training ---------------------------------------- #
    model.train()
    for epoch in tqdm(range(NUM_TRAIN_EPOCHS), desc='Training MVLM', unit='Epoch'):
        logging.info('Start training epoch ' + str(epoch))
        avg_loss, global_step, csv_data = train_epoch(model,
                                            train_dataloader,
                                            optimizer,
                                            device,
                                            global_step,
                                            writers,
                                            csv_data,
                                            pad_token_label_id, 
                                            ACCUMULATION_STEPS, None)
        model.eval()
        with torch.no_grad():
            val_results = eval(model, 
                               tokenizer,
                               val_dataloader, 
                               device, 
                               pad_token_label_id,
                               logging=logging, 
                               print_results=True)

        csv_data['val']['step'].append(global_step)
        csv_data['val']['loss'].append(float(val_results["loss"]))
        csv_data['val']['accuracy'].append(float(val_results["accuracy"]))
        csv_data['val']['perplexity'].append(float(val_results["perplexity"]))
        writer_val.add_scalar("Loss/mvlm", val_results["loss"], global_step=global_step)
        writer_val.add_scalar("Metrics/accuracy", val_results["accuracy"], global_step=global_step)
        writer_val.add_scalar("Metrics/perplexity",val_results["perplexity"], global_step=global_step)
        
        model.train()

        best_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }

    # ---------------------------------------- Save model ---------------------------------------- #
    if SAVE_MODEL_PATH:
        if not os.path.exists(SAVE_MODEL_PATH):
            os.makedirs(SAVE_MODEL_PATH)
        save_path = SAVE_MODEL_PATH + \
                    f'/{log_file[:-4]}_epoch_{best_epoch}.ckpt'
                    
        logging.info('Saving the best model from epoch' + str(best_epoch))
        print('Saving the best model from epoch' + str(best_epoch))
        torch.save(best_state, save_path)
        model.save_pretrained(SAVE_MODEL_PATH)
        print(f"MVLM Model saved in: {SAVE_MODEL_PATH}")

        writer_train.flush()
    writer_train.close()

    # ------------------------------------ Save metrics in CSV ----------------------------------- #
    csv_path = f"{SAVE_MODEL_PATH}/runs"
    os.makedirs(csv_path, exist_ok=True)

    # Save data for 'train' set
    train_csv_file = os.path.join(csv_path, 'train.csv')
    with open(train_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['iteration_number', 'step', 'loss', 'accuracy', 'perplexity'])
        for i, (step, loss, accuracy, perplexity) in enumerate(zip(csv_data['train']['step'],
                                                               csv_data['train']['loss'],
                                                               csv_data['train']['accuracy'],
                                                               csv_data['train']['perplexity'])):
            csv_writer.writerow([i, step, loss, accuracy, perplexity])

    # Save data for 'val' set
    val_csv_file = os.path.join(csv_path, 'val.csv')
    with open(val_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'loss', 'accuracy', 'perplexity'])
        for _, (step, loss, accuracy, perplexity) in enumerate(zip(csv_data['val']['step'], 
                                                             csv_data['val']['loss'],
                                                             csv_data['val']['accuracy'],
                                                             csv_data['val']['perplexity'])):
            csv_writer.writerow([step, loss, accuracy, perplexity])

    # Save data for 'test_tc' set
    test_tc_csv_file = os.path.join(csv_path, 'test_tc.csv')
    with open(test_tc_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'loss', 'f1', 'doc_exact_match'])
        for _, (step, loss, f1, doc_exact_match) in enumerate(zip(csv_data['test_tc']['step'],
                                                                  csv_data['test_tc']['loss'],
                                                                  csv_data['test_tc']['f1'],
                                                                  csv_data['test_tc']['doc_exact_match'])):
            csv_writer.writerow([step, loss, f1, doc_exact_match])

if __name__ == '__main__':
    main()
    