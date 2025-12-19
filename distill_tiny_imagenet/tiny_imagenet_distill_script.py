import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import ViTForImageClassification, ViTImageProcessor, MobileNetV2ForImageClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation alpha')
    
    parser.add_argument('--resume_train', type=str, default=None, help='Path to checkpoint directory')
    parser.add_argument('--resume_epochs_from', type=int, default=0, help='Epoch to resume from if resuming training')
    
    return parser.parse_args()


def plot_curves(epochs, train_losses, val_accuracies, val_f1s):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, color='green', label='Accuracy')
    plt.plot(range(1, epochs + 1), val_f1s, color='orange', label='F1 Score')
    plt.title('Validation Accuracy and F1')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_curves_{epochs}.png')

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_id = 'google/vit-base-patch16-224'
    teacher_model = ViTForImageClassification.from_pretrained(teacher_id)
    teacher_model.to(device)
    teacher_model.eval()

    for param in teacher_model.parameters():
        param.requires_grad = False

    processor = ViTImageProcessor.from_pretrained(teacher_id)

    if args.resume_train:
        student_model = MobileNetV2ForImageClassification.from_pretrained(args.resume_train)
        processor = ViTImageProcessor.from_pretrained(args.resume_train)
    else:
        student_model = MobileNetV2ForImageClassification.from_pretrained('google/mobilenet_v2_1.0_224')
        if student_model.config.num_labels == 1001:
            print('changing to 1000 classes')
            
            old_linear = student_model.classifier
            new_linear = torch.nn.Linear(old_linear.in_features, 1000)
            
            with torch.no_grad():
                new_linear.weight.copy_(old_linear.weight[1:])
                new_linear.bias.copy_(old_linear.bias[1:])

            student_model.classifier = new_linear
            student_model.config.num_labels = 1000

    student_model.to(device)
    print('Done loading models')

    dataset = load_dataset('zh-plus/tiny-imagenet')
    with open('imagenet_class_index.json', 'r') as f:
        imagenet_index_data = json.load(f)

    wnid_to_idx = {v[0]: int(k) for k, v in imagenet_index_data.items()}

    tiny_imagenet_wnids = dataset['train'].features['label'].names

    # model.config.id2label has every class shifted up by 1, 
    # but we changed the linear layer to account for this
    tiny_to_vit_idx = []
    for i, wnid in enumerate(tiny_imagenet_wnids):
        if wnid in wnid_to_idx:
            tiny_to_vit_idx.append(wnid_to_idx[wnid])
        else:
            print(wnid, 'not in map')
            tiny_to_vit_idx.append(-1)

    mapping = torch.tensor(tiny_to_vit_idx).to(device)

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    def train_transforms_fn(examples):
        examples['pixel_values'] = [train_transforms(img.convert('RGB')) for img in examples['image']]
        del examples['image']
        return examples

    def val_transforms_fn(examples):
        examples['pixel_values'] = [val_transforms(img.convert('RGB')) for img in examples['image']]
        del examples['image']
        return examples

    dataset['train'].set_transform(train_transforms_fn)
    dataset['valid'].set_transform(val_transforms_fn)

    train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset['valid'], batch_size=args.batch_size, shuffle=False, num_workers=2)

    print('Done loading data')

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=0.0001)
    if args.resume_train:
        opt_path = os.path.join(args.resume_train, 'optimizer.pt')
        if os.path.exists(opt_path):
            checkpoint_state = torch.load(opt_path, map_location=device)
            optimizer.load_state_dict(checkpoint_state)

    def distill_step(pixel_values, tiny_labels):
        pixel_values = pixel_values.to(device)
        tiny_labels = tiny_labels.to(device)

        imagenet_labels = mapping[tiny_labels]

        student_model.train()
        
        with torch.no_grad():
            teacher_logits = teacher_model(pixel_values).logits

        student_logits = student_model(pixel_values).logits

        loss_ce = F.cross_entropy(student_logits, imagenet_labels, ignore_index=-1) #ignore unmapped classes

        loss_kl = F.kl_div(
            F.log_softmax(student_logits / args.temperature, dim=-1),
            F.softmax(teacher_logits / args.temperature, dim=-1),
            reduction='batchmean'
        ) * (args.temperature ** 2)

        loss = (1 - args.alpha) * loss_ce + args.alpha * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate(model, dataloader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device) #0-199

                outputs = model(pixel_values)   
                preds = torch.argmax(outputs.logits, dim=-1) #1-1000

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true_tiny = torch.tensor(all_labels)
        y_true_full = mapping.cpu()[y_true_tiny].numpy()
        
        y_pred = np.array(all_preds)

        valid_mask = y_true_full != -1
        y_true_valid = y_true_full[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        f1 = f1_score(y_true_valid, y_pred_valid, average='weighted')
        
        return accuracy, f1

    train_losses = []
    val_accuracies = []
    val_f1s = []
    for epoch in range(args.resume_epochs_from, args.epochs + args.resume_epochs_from):
        total_loss = 0.0
        steps = len(train_loader)

        for step, batch in tqdm(enumerate(train_loader), total=steps):
            pixel_values = batch['pixel_values']
            labels = batch['label']

            loss = distill_step(pixel_values, labels)
            total_loss += loss

            if step % 100 == 0:
                print(f'Epoch {epoch+1} [{step+1}/{steps}], Loss: {loss:.4f}')

        avg_loss = total_loss / steps
        train_losses.append(avg_loss)
        
        val_acc, val_f1 = evaluate(student_model, val_loader)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        if (epoch + 1) % 5 == 0:
            checkpoint_dir = f'tiny_imagenet_checkpoint_epoch_{epoch+1}'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            student_model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))

            print(f'checkpoint saved to {checkpoint_dir}')

            plot_curves(epoch + 1, train_losses, val_accuracies, val_f1s)
        
        print('---------------------------------------')
        print(f'epoch {epoch+1} done:')
        print(f'avg train loss: {avg_loss:.4f}')
        print(f'val acc: {val_acc:.4f}')
        print(f'val f1:  {val_f1:.4f}')
        print('---------------------------------------')



    output_dir = f'tiny_imagenet_checkpoint_{args.epochs}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    student_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f'model saved to {output_dir}')

    plot_curves(args.epochs, train_losses, val_accuracies, val_f1s)

if __name__ == '__main__':
    main()