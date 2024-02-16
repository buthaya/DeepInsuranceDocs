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

from torchcrf import CRF

import torch

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, LayoutLMForTokenClassification, LayoutLMTokenizer

sys.path[0] = ''  # nopep8

from deepinsurancedocs.data_preparation.data_utils import label_dict_transform  # nopep8
from deepinsurancedocs.data_preparation.layoutlm_dataset import LayoutLMDataset  # nopep8
from models.layoutlm.prepare_data import LayoutLMDataPreparation  # nopep8
from models.layoutlm.train_utils import LayoutLMWithCRF, eval, train_epoch, eval_crf, train_epoch_crf, layoutlm_collate_fn  # nopep8


def main():
    # ------------------------------- Training args ------------------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='config/payslips_config_ft.json')
    # parser.add_argument('--pretrained_model', type=str,
    #                     default='microsoft/layoutlm-base-uncased')
    # parser.add_argument('--batch_size', type=int, default=2)
    # parser.add_argument('--epoch_num', type=int, default=5)
    # parser.add_argument('--early_stop', type=int, default=3)
    # parser.add_argument('--learning_rate', type=float, default=5e-5)
    # parser.add_argument('--data_ag_file', type=str, default=None)
    current_date = datetime.now().strftime("%d-%m-%Y_%Hh%M")

    args = parser.parse_args()

    config_path = args.config_path

    # Debug parameters
    config_path = 'config/payslips_config_ft_payslips.json'
    # pretrained_model = 'microsoft/layoutlm-base-uncased'
    # batch_size = 1
    # num_train_epochs = 2
    # learning_rate = 5e-1

    model_name = "layoutlm_ft_training"
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
    pretrained_model = config.get('pretrained_model', '')
    checkpoint_path = config.get('checkpoint_path', None)
    batch_size = config['training_parameters'].get('batch_size', 1)
    learning_rate = config['training_parameters'].get('learning_rate', 0.001)
    num_train_epochs = config['training_parameters'].get('epoch_num', 1)
    local_files_only = config.get('is_local_model', False)
    accum_iter = config['training_parameters'].get('gradient_accumulation_steps', 0)
    tagging_scheme = config['preprocessing']['tagging_scheme']

    save_model_path = f'/domino/datasets/local/DeepInsuranceDocs/models/{model_name}/outputs/{dataset_name}_{current_date}'
    os.makedirs(save_model_path, exist_ok=True)
    print(save_model_path)
    # Save the config in the output_dir for auditability
    with open(os.path.join(save_model_path, 'training_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f)
    # ------------------------------------- Tokenizer -------------------------------------- #
    tokenizer = LayoutLMTokenizer.from_pretrained(pretrained_model_name_or_path='microsoft/layoutlm-base-uncased')
    idx2label = label_dict_transform(label_dict=config['label_list'], 
                                     scheme=tagging_scheme)
    label2idx = {label: idx for idx, label in idx2label.items()}
    LABEL_LIST = list(idx2label.values())
    num_labels = len(LABEL_LIST)

    # ---------------------------------------- Data ---------------------------------------- #
    train_dataset = LayoutLMDataset(data_dir, tokenizer, LABEL_LIST, pad_token_label_id, 'test', tagging_scheme)
    test_dataset = LayoutLMDataset(data_dir, tokenizer, LABEL_LIST, pad_token_label_id, 'test', tagging_scheme)

    train_sampler, test_sampler = RandomSampler(
        train_dataset), SequentialSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size,
                                #   collate_fn=layoutlm_collate_fn
                                  )
    test_dataloader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 batch_size=batch_size,
                                #  collate_fn=layoutlm_collate_fn
                                 )

    # ---------------------------------------- Model --------------------------------------- #
    model0 = LayoutLMForTokenClassification.from_pretrained(pretrained_model, num_labels=num_labels)
    model = LayoutLMWithCRF(config= model0.config)
    model = model.to(device)
    crf = model.crf

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_epoch = num_train_epochs
    global_step = 0

    # --------------------------------------- Logging -------------------------------------- #
    if not os.path.exists(save_model_path + '/outputs/logs/'):
        # If it doesn't exist, create it
        os.makedirs(save_model_path + '/outputs/logs/')
    log_file = f'train_layoutlm_{dataset_name}_{current_date}.log'
    logging.basicConfig(filename=save_model_path + '/outputs/logs/' + log_file,
                        level=logging.INFO)
    logging.info('Training settings')
    logging.info(args)

    writer_train = SummaryWriter(
        log_dir=f"{save_model_path}/outputs/runs/{log_file[6:]}/train")
    writer_test = SummaryWriter(
        log_dir=f"{save_model_path}/outputs/runs/{log_file[6:]}/test")
    writers = {'train': writer_train, 'test': writer_test}

    csv_data = {'train': [], 'test': [],
                'f1-score': [], 'precision': [], 'recall': []}

    # --------------------------------------- Training ------------------------------------- #
    for epoch in tqdm(range(num_train_epochs), desc='Training', unit='Epoch'):
        logging.info('Start training epoch ' + str(epoch))
        avg_loss, _ = train_epoch(model, train_dataloader, LABEL_LIST, optimizer, device, global_step, writers, csv_data)
        
        with torch.no_grad():
            avg_tloss = eval_crf(model, test_dataloader, device, pad_token_label_id,
                             idx2label, logging=logging, print_results=False)
            writer_test.add_scalar(
                "Loss/check", avg_tloss, global_step=global_step)
            csv_data['test'].append((epoch, float(avg_loss)))
        best_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }

    # ------------------------------------- Save model ------------------------------------- #
    if save_model_path:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        save_path = save_model_path + \
            f'/{log_file[:-4]}_epoch_{best_epoch}.ckpt'
        logging.info('Saving the best model from epoch' + str(best_epoch))
        print('Saving the best model from epoch' + str(best_epoch))
        torch.save(best_state, save_path)
        model.save_pretrained(save_model_path)

        writer_train.flush()
        writer_test.flush()
    writer_train.close()
    writer_test.close()

    # ------------------------------------- Evaluation ------------------------------------- #
    model.eval()
    eval_crf(model, test_dataloader, device, pad_token_label_id, idx2label, crf, logging=logging)

    # --------------------------------- Save metrics in CSV -------------------------------- #
    for mode in ['train', 'test']:
        csv_path = f"{save_model_path}/outputs/runs//{log_file[6:]}/{mode}/loss.csv"
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            for iteration_number, loss_value in csv_data[mode]:
                # Write the header row to the CSV file
                csv_writer.writerow([iteration_number, loss_value])

    for metric in ['f1-score', 'precision', 'recall']:
        csv_path = f"{save_model_path}/outputs/runs//{log_file[6:]}/train/{metric}.csv"
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            for iteration_number, metric_value in csv_data[metric]:
                # Write the header row to the CSV file
                csv_writer.writerow([iteration_number, metric_value])


if __name__ == '__main__':
    main()
