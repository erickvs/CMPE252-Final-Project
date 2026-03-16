import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging

log = logging.getLogger(__name__)

def train_and_evaluate_dl(model, trainloader, testloader, device, config, callbacks=None):
    """Executes the PyTorch training loop, reporting to callbacks."""
    if callbacks is None:
        callbacks = []
        
    model = model.to(device)
    
    epochs = config.get("epochs", 5)
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 1e-4)
    accumulation_steps = config.get("accumulation_steps", 1)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if config.get("scheduler") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None
        
    log.info(f"Starting DL training on '{device}' for {epochs} epochs...")
    
    for cb in callbacks: cb.on_train_begin(epochs, len(trainloader))
    
    val_acc = 0.0
    for epoch in range(1, epochs + 1):
        for cb in callbacks: cb.on_epoch_begin(epoch)
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            batch_loss = loss.item() * accumulation_steps
            running_loss += batch_loss
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for cb in callbacks: cb.on_batch_end(i, batch_loss)
                
        if scheduler:
            scheduler.step()
            
        train_acc = 100. * correct / total
        avg_train_loss = running_loss / len(trainloader)
        
        # Validation pass
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_running_loss / len(testloader)
        
        metrics = {
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch_time": time.time() - start_time
        }
        
        for cb in callbacks: cb.on_epoch_end(epoch, metrics)
            
    for cb in callbacks: cb.on_train_end(test_accuracy=val_acc)
    
    return val_acc