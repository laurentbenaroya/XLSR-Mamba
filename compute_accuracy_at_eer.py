import argparse
import os

import librosa

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Model
import numpy as np
import csv
from metrics import compute_accuracy_with_eer_threshold

def load_metadata(csv_path):
    label_dict = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file,delimiter=' ')
        for row in reader:
            # filename, _, label = row  # Ignore the middle column (speaker name)
            filename = row[1]
            label = row[5]
            label_dict[filename] = label
    return label_dict


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, bonafide_dir, spoofed_dir, output_file=None):

        self.bonafide_dir = bonafide_dir
        self.spoofed_dir = spoofed_dir

        self.file_bonafide = [os.path.join(bonafide_dir, f) for f in os.listdir(bonafide_dir) if f.endswith('.wav')]
        self.bonafide_labels = ['bonafide'] * len(self.file_bonafide)
        
        self.file_spoofed = [os.path.join(spoofed_dir, f) for f in os.listdir(spoofed_dir) if f.endswith('.wav')]
        self.spoofed_labels = ['spoofed'] * len(self.file_spoofed)
        self.file_list = self.file_bonafide + self.file_spoofed
        self.labels = self.bonafide_labels + self.spoofed_labels
        if output_file:
            with open(output_file, 'w') as f:
                for file, label in zip(self.file_list, self.labels):
                    f.write(f"{file},{label}\n")
                f.close()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")        
        data, sr = librosa.load(file_path, sr=16000)
        label = self.labels[idx]
        return data, label, file_path

### collate class that pads the input sequences
class Collate_Eval:
    def __init__(self):
        pass

    def __call__(self, batch):
        batch_x = []
        label_x = []
        utt_id = []
        for data, label, file_path in batch:
            audio = data
            batch_x.append(audio)
            label_x.append(1 if label == 'bonafide' else 0)  # Convert label to binary
            utt_id.append(os.path.basename(file_path))  # Use the filename as the ID
        if len(batch_x) == 1:
            # If there's only one item in the batch, we need to convert it to a tensor
            batch_x = torch.tensor(batch_x[0], dtype=torch.float32).unsqueeze(0)
        else:
            # Pad the sequences to the maximum length in the batch
            # Convert each audio sequence to a tensor and pad them
            batch_x = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in batch_x], batch_first=True, padding_value=0.0)
        return batch_x, torch.tensor(label_x), utt_id

def produce_evaluation_file(bonafide_dir, spoofed_dir, model, device, save_path):

    dataset = EvaluationDataset(bonafide_dir, spoofed_dir)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=Collate_Eval())
    model.eval()
    model.to(device)
    fname_list = []
    score_list = []
    label_list = []     
    with torch.no_grad():
        for batch_x, label_x, utt_id in tqdm(data_loader,total=len(data_loader)):

            batch_x = batch_x.to(device)  
            batch_out = model(batch_x)
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel() 
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            label_list.extend(label_x)
            
    with open(save_path, 'a+') as fh:
        for f, cm, lab in zip(fname_list,score_list,label_list):
            fh.write('{} {} {}\n'.format(f, cm, lab))
    fh.close()   
    print('Scores saved to {}'.format(save_path))
    accuracy, cm_eer_threshold, precision, recall, f1_score = compute_accuracy_with_eer_threshold(np.array(label_list),
                                                            np.array(score_list), all_metrics=True)
    print("CM threshold: ", cm_eer_threshold)
    print("CM accuracy: ", accuracy)
    print("CM precision: ", precision)
    print("CM recall: ", recall)
    print("CM f1_score: ", f1_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute accuracy at EER")
    parser.add_argument("--bonafidedir", type=str, required=True, help='Path to bonafide audio files')
    parser.add_argument("--spoofeddir", type=str, required=True, help='Path to spoofed audio files')
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the evaluation results")
    #model parameters
    parser.add_argument('--emb-size', type=int, default=144, metavar='N', help='embedding size of the model')
    parser.add_argument('--num_encoders', type=int, default=12, metavar='N', help='number of encoders of the mamba blocks')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')    
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
      
    args = parser.parse_args()

    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Produce evaluation file
    produce_evaluation_file(args.bonafidedir, args.spoofeddir, model, device, args.output_file)
