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

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, LayoutLMForMaskedLM

from data.PAYSLIPS.config import LABEL_LIST

sys.path[0] = ''  # nopep8

from deepinsurancedocs.data_preparation.data_utils import label_dict_transform  # nopep8
from models.layoutlm.prepare_data_mvlm import LayoutLMDataPreparationMVLM  # nopep8
from models.layoutlm.train_utils_mvlm import eval, train_epoch, layoutlm_collate_fn  # nopep8


def main():
    # ------------------------------- Training args ------------------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='config/funsd_config.json')
    parser.add_argument('--pretrained_model', type=str,
                        default='microsoft/layoutlm-base-uncased')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epoch_num', type=int, default=5)
    # parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    # parser.add_argument('--data_ag_file', type=str, default=None)
    current_date = datetime.now().strftime("%d-%m-%Y_%Hh%M")

    args = parser.parse_args()

    config_path = args.config_path
    pretrained_model = args.pretrained_model
    batch_size = args.batch_size
    num_train_epochs = args.epoch_num

    # Debug parameters
    config_path = 'config/funsd_config.json'
    pretrained_model = 'microsoft/layoutlm-base-uncased'
    batch_size = 1
    num_train_epochs = 2
    learning_rate = 5e-5

    model_name = "layoutlm_mvlm_training"
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # ------------------------ Open Config with dataset information ------------------------ #
    with open(config_path, 'r', encoding='utf-8') as f:
        # config is a dict to store the following information about the dataset:
        # - data_dir: path to the directory containing the dataset
        # - input_format: format of the input data
        # - preprocessing: dictionary containing the tagging scheme used for preprocessing
        # - label_list: dictionary containing the mapping of labels to their corresponding indices
        config = json.load(f)
    dataset_name = config['data_dir'].split('/')[-1]
    save_model_path = f'models/{model_name}/outputs/{dataset_name}_{current_date}'
    os.makedirs(save_model_path, exist_ok=True)

    # ---------------------------------------- Data ---------------------------------------- #
    dataset = LayoutLMDataPreparationMVLM(config_path, mask_proba=0.15, mask_label_token_id=pad_token_label_id)

    dataset_dict = dataset.load_training_ready_dataset()
    train_dataset = dataset_dict['train']
    test_dataset = dataset_dict['test']

    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size,
                                  collate_fn=layoutlm_collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 batch_size=batch_size,
                                 collate_fn=layoutlm_collate_fn)

    # ---------------------------------------- Model --------------------------------------- #
    model = LayoutLMForMaskedLM.from_pretrained(pretrained_model)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_epoch = num_train_epochs
    global_step = 0


    # --------------------------------------- Logging -------------------------------------- #
    if not os.path.exists(save_model_path + '/outputs/logs/'):
        # If it doesn't exist, create it
        os.makedirs(save_model_path + '/outputs/logs/')
    log_file = f'train_{model_name}_{dataset_name}_{current_date}.log'
    logging.basicConfig(filename=save_model_path + '/outputs/logs/' + log_file,
                        level=logging.INFO)
    logging.info('Training settings')
    logging.info(args)

    writer_train = SummaryWriter(
        log_dir=f"{save_model_path}/outputs/runs/{log_file[6:]}/train")
    writer_test = SummaryWriter(
        log_dir=f"{save_model_path}/outputs/runs/{log_file[6:]}/test")
    writers = {'train': writer_train, 'test': writer_test}

    csv_data = {'train': [], 'test': [], 'accuracy': []}

    # --------------------------------------- Training ------------------------------------- #
    for epoch in range(num_train_epochs):
        logging.info('Start training epoch ' + str(epoch))
        avg_loss, global_step = train_epoch(model,
                                            train_dataloader,
                                            optimizer,
                                            device,
                                            global_step,
                                            writers,
                                            csv_data,
                                            pad_token_label_id)
        with torch.no_grad():
            avg_tloss = eval(model,
                             test_dataloader,
                             device,
                             pad_token_label_id,
                             logging=logging,
                             print_results=False)
            writer_test.add_scalar("Loss/check", avg_tloss, global_step=global_step)
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
                    f'/{args.log_file[:-4]}_epoch_{best_epoch}.ckpt'
        logging.info('Saving the best model from epoch' + str(best_epoch))
        print('Saving the best model from epoch' + str(best_epoch))
        torch.save(best_state, save_path)

        writer_train.flush()
        writer_test.flush()
    writer_train.close()
    writer_test.close()

    # ------------------------------------- Evaluation ------------------------------------- #
    model.eval()
    eval(model, test_dataloader, device,
              pad_token_label_id, logging=logging)

    # --------------------------------- Save metrics in CSV -------------------------------- #
    for mode in ['train', 'test']:
        csv_path = f"runs/{args.log_file[6:]}/{mode}/loss.csv"
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            for iteration_number, loss_value in csv_data[mode]:
                # Write the header row to the CSV file
                csv_writer.writerow([iteration_number, loss_value])

    for metric in ['accuracy']:
        csv_path = f"runs/{args.log_file[6:]}/train/{metric}.csv"
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            for iteration_number, metric_value in csv_data[metric]:
                # Write the header row to the CSV file
                csv_writer.writerow([iteration_number, metric_value])


if __name__ == '__main__':
    main()
