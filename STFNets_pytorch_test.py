import numpy as np
import torch
from torch import nn
from widar_dataset import WiDARDataset
from STFNets_pytorch import STFNet, SERIES_SIZE
from torch.utils.data import DataLoader
import click
import tqdm
from sklearn.metrics import f1_score


@click.command()
@click.option('--model-path', type=str, default='best_model.pth', help='Path to the trained model')
@click.option('--data-path', type=str, default='data/', help='Path to the dataset')
@click.option('--additive-noise', type=float, default=1.0, help='Standard deviation of additive noise')
@click.option('--additive-noise-ratio', type=float, default=0.0, help='Ratio of additive noise to the signal')
@click.option('--batch-size', type=int, default=64, help='Batch size for testing')
@click.option('--num-workers', type=int, default=4, help='Number of workers for data loading')
def test_stfnets(**argv):
    model_path = argv.get('model_path')
    data_path = argv.get('data_path')
    additive_noise = argv.get('additive_noise')
    additive_noise_ratio = argv.get('additive_noise_ratio')
    batch_size = argv.get('batch_size')
    num_workers = argv.get('num_workers')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.inference_mode():
        # train_dataset = NPZDataset(train_npz_path, SERIES_SIZE, SENSOR_AXIS, SENSOR_NUM)
        must_have=[f'-gesture{i}-' for i in range(4)]+['-gesture17-','-gesture18-']
        # must_have=[f'-user10-gesture{i}-' for i in range(4)]+['-user10-gesture17-','-user10-gesture18-']
        # must_not_have=["-user5-",]
        series_size = SERIES_SIZE
        test_dataset = WiDARDataset(data_path, target_size=series_size, min_data_len=0, split_ratio=1.0, must_have=must_have, additive_noise_ratio=additive_noise_ratio, additive_noise_std=additive_noise)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=4)
        
        # Initialize Model
        model = STFNet(input_size=series_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Loss Function
        criterion = nn.CrossEntropyLoss()

        # Testing Loop
        test_loss = 0
        correct = 0
        total = 0
        total_labels = []
        total_preds = []


        all_preds = []
        all_labels = []
        with tqdm.tqdm(test_loader, desc="Evaluating") as test_t:
            for test_data, test_target in test_t:
                test_data, test_target = test_data.to(device), test_target.to(device)
                test_output = model(test_data)
                
                test_target_indices = torch.argmax(test_target, dim=1)
                batch_loss = criterion(test_output, test_target_indices)
                test_loss += batch_loss.item()
                
                pred = test_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(test_target_indices.view_as(pred)).sum().item()
                total += test_data.size(0)
                
                total_labels.extend(test_target_indices.cpu().numpy())
                total_preds.extend(pred.cpu().numpy().flatten())

        dev_accuracy = correct / total
        dev_cross_entropy = test_loss / len(test_loader)
        dev_macro_f1 = f1_score(total_labels, total_preds, average='macro')

        print(f"Test Loss: {dev_cross_entropy:.4f}, Test Accuracy: {dev_accuracy:.4f}, Test Macro F1 Score: {dev_macro_f1:.4f}")

if __name__ == '__main__':
    test_stfnets()

