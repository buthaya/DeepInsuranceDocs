'''
train_no_pos.py
script:
    - "python train.py --data_dir inputs/processed_data --save_path models --batch_size 16 --epoch_num 1 --early_stop 3 --learning_rate 5e-5 --data_ag_file inputs/data_augmentation/dataset_anonymized.json --log_file example.log"
 
Train the model, either from a pre-trained model either from a saved checkpoint.
'''
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from tqdm import tqdm

import torch

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, LayoutLMForTokenClassification, LayoutLMTokenizer, LayoutLMConfig

sys.path[0] = ''  # nopep8

from deepinsurancedocs.data_preparation.data_utils import label_dict_transform  # nopep8
from deepinsurancedocs.data_preparation.layoutlm_dataset import LayoutLMDataset  # nopep8
from models.layoutlm.model import LayoutLMForTokenClassificationInternal
from models.layoutlm.prepare_data import LayoutLMDataPreparation  # nopep8
from models.layoutlm.train_utils import eval, train_epoch, layoutlm_collate_fn  # nopep8

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def train_token_classifier(config_path, 
                           tokenizer=None,
                           model_name="layoutlm_tc",
                           pretrained_model=None,
                           print_batch=False,
                           save=True
                           ):
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index
    # ------------------ Open Config with dataset & training information ------------------ #
    with open(config_path, 'r', encoding='utf-8') as f:
        # config is a dict to store the following information about the dataset:
        # - data_dir: path to the directory containing the dataset
        # - input_format: format of the input data
        # - preprocessing: dictionary containing the tagging scheme used for preprocessing
        # - label_list: dictionary containing the mapping of labels to their corresponding indices
        config = json.load(f)

    data_dir = config.get('data_dir', '')
    dataset_name = data_dir.split('/')[-1]
    # If we don't mention a pretrained model in the function, load the default one from the config
    # This is useful so that we can load weights from the pretrained model during
    # MVLM training
    if not pretrained_model:
        pretrained_model = config.get('pretrained_model', None)
    checkpoint_path = config.get('checkpoint_path', None)
    batch_size = config['training_parameters'].get('batch_size', 1)
    learning_rate = config['training_parameters'].get('learning_rate', 0.001)
    num_train_epochs = config['training_parameters'].get('epoch_num', 1)
    local_files_only = config.get('is_local_model', False)
    accumulation_steps = config['training_parameters'].get('gradient_accumulation_steps', 1)
    tagging_scheme = config['preprocessing']['tagging_scheme']
    MODEL_DIR = config.get('model_dir', None)
    current_date = datetime.now().strftime("%d-%m-%Y_%Hh%M")

    save_model_path = f'/domino/datasets/local/DeepInsuranceDocs/models/{model_name}/{dataset_name}/{current_date}'
    os.makedirs(save_model_path, exist_ok=True)
    print(f"TC Training date: {current_date}")
    print(f"TC Model saved in: {save_model_path}")
    # Save the config in the output_dir for auditability
    with open(os.path.join(save_model_path, 'training_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f)
    # ------------------------------------- Tokenizer -------------------------------------- #
    tokenizer = LayoutLMTokenizer.from_pretrained(os.path.join(MODEL_DIR, 'microsoft/layoutlm-base-uncased'))
    
    idx2label = label_dict_transform(label_dict=config['label_list'], 
                                     scheme=tagging_scheme)
    label2idx = {label: idx for idx, label in idx2label.items()}
    LABEL_LIST = list(idx2label.values())
    num_labels = len(LABEL_LIST)

    # ---------------------------------------- Data ---------------------------------------- #
    train_dataset = LayoutLMDataset(data_dir, tokenizer, LABEL_LIST, pad_token_label_id, 'train', tagging_scheme)
    val_dataset = LayoutLMDataset(data_dir, tokenizer, LABEL_LIST, pad_token_label_id, 'test', tagging_scheme)

    train_sampler, val_sampler = RandomSampler(
        train_dataset), SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size,
                                #   collate_fn=layoutlm_collate_fn
                                  num_workers=6,
                                  )
    val_dataloader = DataLoader(val_dataset,
                                 sampler=val_sampler,
                                 batch_size=batch_size,
                                #  collate_fn=layoutlm_collate_fn
                                  num_workers=6,
                                 )

    # ---------------------------------------- Model --------------------------------------- #
    # Create your own classification model with crossentropyloss by ignoring the 

    if pretrained_model:
        model = LayoutLMForTokenClassificationInternal.from_pretrained(pretrained_model, num_labels=num_labels)
    else:
        model = LayoutLMForTokenClassificationInternal(LayoutLMConfig(num_labels = num_labels))
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_epoch = num_train_epochs
    global_step = 0

    # --------------------------------------- Logging -------------------------------------- #
    if not os.path.exists(save_model_path + '/logs/'):
        # If it doesn't exist, create it
        os.makedirs(save_model_path + '/logs/')
    log_file = f'train_layoutlm_{dataset_name}_{current_date}.log'
    logging.basicConfig(filename=save_model_path + '/logs/' + log_file,
                        level=logging.INFO)
    logging.info('Training settings')

    writer_train = SummaryWriter(
        log_dir=f"{save_model_path}/tc_runs/train")
    writer_val = SummaryWriter(
        log_dir=f"{save_model_path}/tc_runs/val")
    writers = {'train': writer_train, 'val': writer_val}

    csv_data = {'train': {'step' : [],
                          'loss': [],
                          'precision': [], 
                          'recall': [],
                          'f1': []}, 
                'val': {'step' : [],
                        'loss': [],
                        'precision': [], 
                        'recall': [],
                        'f1': [],
                        'doc_exact_match': []}}

    # ---------------------------- Eval at the start of training ---------------------------- #
    model.eval()
    with torch.no_grad():
        eval_dict = eval(model, val_dataloader, device, pad_token_label_id,
                            idx2label, logging=logging, print_results=False)


        csv_data['val']['step'].append(global_step)
        csv_data['val']['loss'].append(float(eval_dict['loss']))
        csv_data['val']['precision'].append(float(eval_dict['precision']))
        csv_data['val']['recall'].append(float(eval_dict['recall']))
        csv_data['val']['f1'].append(float(eval_dict['f1']))
        csv_data['val']['doc_exact_match'].append(float(eval_dict['doc_exact_match']))
        writer_val.add_scalar("Loss/check", eval_dict['loss'], global_step=global_step)
        writer_val.add_scalar("Metrics/f1",eval_dict['f1'], global_step=global_step)
        writer_val.add_scalar("Metrics/precision",eval_dict['precision'], global_step=global_step)
        writer_val.add_scalar("Metrics/recall",eval_dict['recall'], global_step=global_step)
        writer_val.add_scalar("Metrics/doc_exact_match",eval_dict['doc_exact_match'], global_step=global_step)
    model.train()
    # --------------------------------------- Training ------------------------------------- #
    for epoch in tqdm(range(num_train_epochs), desc='Training Token Classifier', unit='Epoch'):
        model.train()
        logging.info('Start training epoch ' + str(epoch))
        avg_loss, global_step = train_epoch(
            model, train_dataloader, LABEL_LIST, optimizer, device, global_step, writers, csv_data,
            print_batch, accumulation_steps, None)

        
        # ---------------------------- Eval at the end of epoch ---------------------------- #
        model.eval()
        with torch.no_grad():
            eval_dict = eval(model, val_dataloader, device, pad_token_label_id,
                             idx2label, logging=logging, print_results=False)


            csv_data['val']['step'].append(global_step)
            csv_data['val']['loss'].append(float(eval_dict['loss']))
            csv_data['val']['precision'].append(float(eval_dict['precision']))
            csv_data['val']['recall'].append(float(eval_dict['recall']))
            csv_data['val']['f1'].append(float(eval_dict['f1']))
            csv_data['val']['doc_exact_match'].append(float(eval_dict['doc_exact_match']))
            writer_val.add_scalar("Loss/check", eval_dict['loss'], global_step=global_step)
            writer_val.add_scalar("Metrics/f1",eval_dict['f1'], global_step=global_step)
            writer_val.add_scalar("Metrics/precision",eval_dict['precision'], global_step=global_step)
            writer_val.add_scalar("Metrics/recall",eval_dict['recall'], global_step=global_step)
            writer_val.add_scalar("Metrics/doc_exact_match",eval_dict['doc_exact_match'], global_step=global_step)
        model.train()
        # ---------------------------------------------------------------------------------- #

        
        best_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }

    # ------------------------------------- Save model ------------------------------------- #
    if save_model_path and save:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        save_path = save_model_path + \
            f'/{log_file[:-4]}_epoch_{best_epoch}.ckpt'
        logging.info('Saving the best model from epoch' + str(best_epoch))
        print('Saving the best model from epoch' + str(best_epoch))
        torch.save(best_state, save_path)
        model.save_pretrained(save_model_path)

        writer_train.flush()
        writer_val.flush()
    writer_train.close()
    writer_val.close()

    # ------------------------------------- Evaluation ------------------------------------- #
    model.eval()

    # ------------------------------------ Save metrics in CSV ----------------------------------- #
    csv_path = f"{save_model_path}/tc_runs"
    os.makedirs(csv_path, exist_ok=True)

    # Save data for 'train' set
    train_csv_file = os.path.join(csv_path, 'train.csv')
    with open(train_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['iteration_number', 'loss', 'precision', 'recall', 'f1'])
        for _, (step, loss, precision, recall, f1) in enumerate(zip(csv_data['train']['step'],
                                                                    csv_data['train']['loss'],
                                                                    csv_data['train']['precision'],
                                                                    csv_data['train']['recall'],
                                                                    csv_data['train']['f1'])):
            csv_writer.writerow([step, loss, precision, recall, f1])

    # Save data for 'val' set
    val_csv_file = os.path.join(csv_path, 'val.csv')
    with open(val_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['iteration_number', 'loss', 'precision', 'recall', 'f1', 'doc_exact_match'])
        for _, (step, loss, precision, recall, f1, doc_exact_match) in enumerate(zip(csv_data['val']['step'],
                                                                                     csv_data['val']['loss'],
                                                                                     csv_data['val']['precision'],
                                                                                     csv_data['val']['recall'],
                                                                                     csv_data['val']['f1'],
                                                                                     csv_data['val']['doc_exact_match'])):
            csv_writer.writerow([step, loss, precision, recall, f1, doc_exact_match])

    return save_model_path, eval(model, val_dataloader, device,pad_token_label_id, idx2label, logging=logging)


def main():
    # ------------------------------- Training args ------------------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='config/payslips_config_ft_payslips.json')

    args = parser.parse_args()

    config_path = args.config_path

    # Debug parameters
    # config_path = 'config/token_classif_payslips.json'
    # config_path = '/domino/datasets/local/DeepInsuranceDocs/models/layoutlm_full_pipeline/mvlm_docile_10k_tc_docile/05-02-2024_15h29/token_classif_config.json'
    # print(f"CAREFUL !!! DEBUG MODE USING {config_path}")
    # config_path = '/mnt/config/token_classif_docile.json'

    save_model_path, eval_result_dict = train_token_classifier(config_path,
                                                               print_batch=True,
                                                               save=True)


if __name__ == '__main__':
    main()
