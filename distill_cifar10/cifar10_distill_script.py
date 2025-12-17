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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--temperature", type=float, default=3.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Distillation alpha")
    
    parser.add_argument("--resume_train", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--resume_epochs_from", type=int, default=0, help="Epoch to resume from if resuming training")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_id = 'nateraw/vit-base-patch16-224-cifar10'
    teacher_model = ViTForImageClassification.from_pretrained(teacher_id)
    teacher_model.to(device)
    teacher_model.eval()

    for param in teacher_model.parameters():
        param.requires_grad = False

    processor = ViTImageProcessor.from_pretrained(teacher_id)

    if args.resume_train:
        student_model = MobileNetV2ForImageClassification.from_pretrained(args.resume_train)
    else:
        student_model = MobileNetV2ForImageClassification.from_pretrained('google/mobilenet_v2_1.0_224')

        NUM_CLASSES = 10
        student_model.classifier = torch.nn.Linear(student_model.classifier.in_features, NUM_CLASSES)
        student_model.config.num_labels = NUM_CLASSES

    student_model.to(device)

    print('Done loading models')

    dataset = load_dataset('uoft-cs/cifar10')

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
        examples['pixel_values'] = [train_transforms(img.convert('RGB')) for img in examples['img']]
        del examples["img"]
        return examples

    def val_transforms_fn(examples):
        examples['pixel_values'] = [val_transforms(img.convert('RGB')) for img in examples['img']]
        del examples["img"]
        return examples

    dataset['train'].set_transform(train_transforms_fn)
    dataset['test'].set_transform(val_transforms_fn)

    train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False, num_workers=2)

    print('Done loading data')

    # ------------------------------------

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=0.0001)

    def distill_step(pixel_values, labels):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        student_model.train()
        with torch.no_grad():
            teacher_logits = teacher_model(pixel_values).logits

        student_logits = student_model(pixel_values).logits

        loss_ce = F.cross_entropy(student_logits, labels)
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
            for batch in dataloader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                outputs = model(pixel_values)
                preds = torch.argmax(outputs.logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return accuracy_score(all_labels, all_preds)


    train_losses = []
    val_accuracies = []
    for epoch in range(args.resume_epochs_from, args.epochs + args.resume_epochs_from):
        total_loss = 0.0
        steps = len(train_loader)

        for step, batch in tqdm(enumerate(train_loader)):
            pixel_values = batch['pixel_values']
            labels = batch['label']

            loss = distill_step(pixel_values, labels)
            total_loss += loss

            if step % 100 == 0:
                print(f'Epoch {epoch+1} [{step+1}/{steps}], Loss: {loss:.4f}')

        avg_loss = total_loss / steps
        train_losses.append(avg_loss)
        
        val_acc = evaluate(student_model, val_loader)
        val_accuracies.append(val_acc)

        print('---------------------------------------')
        print(f'epoch {epoch+1} done:')
        print(f'avg train loss: {avg_loss:.4f}')
        print(f'val acc: {val_acc:.4f}')
        print('---------------------------------------')




    output_dir = f'cifar10_checkpoint_{args.epochs}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    student_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"modelsaved to {output_dir}")


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), val_accuracies, color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f'training_curves_{args.epochs}.png')


if __name__ == "__main__":
    main()